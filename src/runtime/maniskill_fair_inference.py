from __future__ import annotations
import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import math
import time
import torch
import torch.nn as nn


@dataclass(frozen=True)
class FairRuntimeConfig:
    precision: str = "fp32"
    backend: str = "eager"
    compile_mode: str = "default"
    warmup_calls: int = 10
    include_input_transfer: bool = False

    def validate(self, device: torch.device) -> None:
        if self.precision not in {"fp32", "tf32", "bf16"}:
            raise ValueError("precision must be fp32, tf32, or bf16")
        if self.backend not in {"eager", "compile", "cudagraph"}:
            raise ValueError("backend must be eager, compile, or cudagraph")
        if self.backend == "cudagraph" and device.type != "cuda":
            raise ValueError("CUDA Graph runtime requires CUDA")
        if self.compile_mode not in {"default", "reduce-overhead"}:
            raise ValueError("compile_mode must be default or reduce-overhead")
        if self.warmup_calls < 0:
            raise ValueError("warmup_calls must be non-negative")
        if device.type != "cuda" and self.precision != "fp32":
            raise ValueError("TF32/BF16 fair runtime requires CUDA")

    def to_dict(self) -> dict:
        return asdict(self)


def add_fair_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--runtime-precision", choices=("fp32", "tf32", "bf16"), default="fp32"
    )
    parser.add_argument(
        "--runtime-backend", choices=("eager", "compile", "cudagraph"), default="eager"
    )
    parser.add_argument(
        "--runtime-compile-mode",
        choices=("default", "reduce-overhead"),
        default="default",
    )
    parser.add_argument("--runtime-warmup-calls", type=int, default=10)
    parser.add_argument("--include-input-transfer", action="store_true")


def runtime_from_args(args, device: torch.device) -> FairRuntimeConfig:
    config = FairRuntimeConfig(
        precision=args.runtime_precision,
        backend=args.runtime_backend,
        compile_mode=args.runtime_compile_mode,
        warmup_calls=args.runtime_warmup_calls,
        include_input_transfer=args.include_input_transfer,
    )
    config.validate(device)
    return config


def configure_torch_runtime(config: FairRuntimeConfig, device: torch.device) -> None:
    if device.type != "cuda":
        return
    use_tf32 = config.precision == "tf32"
    torch.backends.cuda.matmul.allow_tf32 = use_tf32
    torch.backends.cudnn.allow_tf32 = use_tf32
    torch.set_float32_matmul_precision("high" if use_tf32 else "highest")


def autocast_context(config: FairRuntimeConfig, device: torch.device):
    if device.type == "cuda" and config.precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def compile_module(module: nn.Module, config: FairRuntimeConfig) -> nn.Module:
    if config.backend in {"eager", "cudagraph"}:
        return module
    if config.compile_mode == "default":
        return torch.compile(
            module, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False}
        )
    return torch.compile(module, mode="reduce-overhead", fullgraph=True, dynamic=False)


class CUDAGraphInferenceWrapper(nn.Module):
    """Capture one frozen, fixed-shape module and replay it without Python launches."""

    def __init__(
        self,
        module: nn.Module,
        sample_args: tuple[torch.Tensor, ...],
        *,
        keyword_order: tuple[str, ...] = (),
    ) -> None:
        super().__init__()
        if not sample_args or any(
            (value.device.type != "cuda" for value in sample_args)
        ):
            raise ValueError("CUDA Graph capture requires non-empty CUDA tensor inputs")
        if any((parameter.requires_grad for parameter in module.parameters())):
            raise ValueError("CUDA Graph inference module must be frozen")
        self.module = module
        self.keyword_order = keyword_order
        if hasattr(module, "config"):
            self.config = module.config
        self.device = sample_args[0].device
        if any((value.device != self.device for value in sample_args)):
            raise ValueError("CUDA Graph inputs must share one device")
        self.static_args = tuple((torch.empty_like(value) for value in sample_args))
        producer = torch.cuda.current_stream(self.device)
        capture_stream = torch.cuda.Stream(device=self.device)
        capture_stream.wait_stream(producer)
        with torch.cuda.stream(capture_stream), torch.inference_mode():
            for static, sample in zip(self.static_args, sample_args):
                static.copy_(sample)
            for _ in range(3):
                module(*self.static_args)
        producer.wait_stream(capture_stream)
        producer.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with (
            torch.cuda.graph(self.graph, stream=capture_stream),
            torch.inference_mode(),
        ):
            self.static_output = module(*self.static_args)
        producer.wait_stream(capture_stream)
        producer.synchronize()

    def forward(self, *args: torch.Tensor, **kwargs: torch.Tensor):
        if kwargs:
            if set(kwargs) != set(self.keyword_order):
                raise ValueError("CUDA Graph call changed its keyword contract")
            args = (*args, *(kwargs[name] for name in self.keyword_order))
        if len(args) != len(self.static_args):
            raise ValueError("CUDA Graph call has the wrong number of inputs")
        for static, value in zip(self.static_args, args):
            if (
                value.shape != static.shape
                or value.dtype != static.dtype
                or value.device != static.device
            ):
                raise ValueError("CUDA Graph call changed a static input contract")
            static.copy_(value)
        self.graph.replay()
        return self.static_output


def graph_module(
    module: nn.Module,
    config: FairRuntimeConfig,
    sample_args: tuple[torch.Tensor, ...],
    *,
    keyword_order: tuple[str, ...] = (),
) -> nn.Module:
    if config.backend != "cudagraph":
        return module
    return CUDAGraphInferenceWrapper(module, sample_args, keyword_order=keyword_order)


class DeviceTimer:
    """Synchronized CUDA-event timer with a CPU wall-clock fallback."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self._cpu_start: float | None = None
        self._start = self._end = None

    def start(self) -> None:
        if self.device.type == "cuda":
            self._start = torch.cuda.Event(enable_timing=True)
            self._end = torch.cuda.Event(enable_timing=True)
            self._start.record()
        else:
            self._cpu_start = time.perf_counter()

    def stop_ms(self) -> float:
        if self.device.type == "cuda":
            assert self._start is not None and self._end is not None
            self._end.record()
            self._end.synchronize()
            return float(self._start.elapsed_time(self._end))
        assert self._cpu_start is not None
        return 1000.0 * (time.perf_counter() - self._cpu_start)


class ObservationBuffer:
    """Persistent batch-one observation tensor used identically by every method."""

    def __init__(self, shape: tuple[int, ...], device: torch.device) -> None:
        self.device = device
        self.tensor = torch.empty(shape, dtype=torch.float32, device=device)

    def update(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape != self.tensor.shape or value.dtype != torch.float32:
            raise ValueError("observation buffer input has the wrong shape or dtype")
        self.tensor.copy_(value, non_blocking=self.device.type == "cuda")
        return self.tensor


def backfill_temporal_aggregation_proposals(
    storage: torch.Tensor,
    proposal_batch: torch.Tensor,
    *,
    first_timestep: int,
    proposal_horizon: int,
) -> None:
    """Insert deferred U-ACT proposals exactly as query-frequency-one TA would.

    Active Flow blocks do not consume the intermediate TA actions, so their
    U-ACT forwards may be batched and replayed immediately before the next
    decision boundary.  Writing each proposal back to its original row keeps
    the next boundary's temporal aggregation bitwise identical to the
    per-environment-step bookkeeping path.
    """
    if storage.ndim != 3 or proposal_batch.ndim != 3:
        raise ValueError("TA storage and proposal batch must both be rank three")
    if proposal_batch.shape[1] != proposal_horizon:
        raise ValueError("deferred proposal horizon changed")
    if proposal_batch.shape[2] != storage.shape[2]:
        raise ValueError("deferred proposal action dimension changed")
    if first_timestep < 0:
        raise ValueError("deferred proposal block starts before timestep zero")
    last_timestep = first_timestep + proposal_batch.shape[0] - 1
    if last_timestep >= storage.shape[0]:
        raise ValueError("deferred proposal block exceeds TA storage rows")
    if last_timestep + proposal_horizon > storage.shape[1]:
        raise ValueError("deferred proposal block exceeds TA storage columns")
    for offset in range(proposal_batch.shape[0]):
        row = first_timestep + offset
        storage[row, row : row + proposal_horizon].copy_(proposal_batch[offset])


class DualStreamUACTRunner:
    """Overlap independent deferred-mean and current-decision U-ACT forwards.

    The two inputs have already been copied by the producer stream.  Both
    consumers only read frozen parameters, and the producer waits for both
    outputs before temporal aggregation or routing can observe them.
    """

    def __init__(self, device: torch.device) -> None:
        if device.type != "cuda":
            raise ValueError("dual-stream U-ACT execution requires CUDA")
        self.device = device
        self.deferred_stream = torch.cuda.Stream(device=device)
        self.decision_stream = torch.cuda.Stream(device=device)

    def run(
        self,
        deferred_module: nn.Module,
        decision_module: nn.Module,
        deferred_observations: torch.Tensor,
        decision_observations: torch.Tensor,
        runtime: FairRuntimeConfig,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        producer = torch.cuda.current_stream(self.device)
        self.deferred_stream.wait_stream(producer)
        self.decision_stream.wait_stream(producer)
        with torch.cuda.stream(self.deferred_stream):
            with autocast_context(runtime, self.device):
                proposal_batch = deferred_module(deferred_observations)
        with torch.cuda.stream(self.decision_stream):
            with autocast_context(runtime, self.device):
                proposal_mean, uncertainty = decision_module(decision_observations)
        producer.wait_stream(self.deferred_stream)
        producer.wait_stream(self.decision_stream)
        return (proposal_batch, proposal_mean, uncertainty)


class FixedStepRefiner(nn.Module):
    """Compile-friendly wrapper around a frozen action Flow at one fixed C."""

    def __init__(self, flow: nn.Module, steps: int) -> None:
        super().__init__()
        if steps <= 0:
            raise ValueError("fixed Flow steps must be positive")
        self.flow = flow
        self.steps = steps
        self.register_buffer(
            "times",
            torch.arange(steps, dtype=torch.float32, device=flow.action_mean.device)
            / steps,
        )

    def forward(
        self, observations: torch.Tensor, source_chunk: torch.Tensor
    ) -> torch.Tensor:
        arm_dim = self.flow.config.arm_dim
        source_arm = (
            source_chunk[..., :arm_dim] - self.flow.action_mean[:arm_dim]
        ) / self.flow.action_std[:arm_dim]
        normalized_observations = (
            observations - self.flow.obs_mean
        ) / self.flow.obs_std
        static_condition = self.flow.velocity_net.encode_static_condition(
            normalized_observations, source_arm
        )
        sample = source_arm
        for index in range(self.steps):
            time = self.times[index].to(dtype=sample.dtype).expand(sample.shape[0])
            velocity = self.flow.velocity_net.forward_with_static_condition(
                sample, time, static_condition
            )
            sample = sample + velocity / self.steps
        refined_arm = (
            sample * self.flow.action_std[:arm_dim] + self.flow.action_mean[:arm_dim]
        )
        return torch.cat([refined_arm, source_chunk[..., arm_dim:]], dim=-1)


def _cached_first_decoder_layer(
    decoder: nn.TransformerDecoder,
    cached_after_self_attention: torch.Tensor,
    memory: torch.Tensor,
) -> torch.Tensor:
    """Resume a norm-first decoder after its observation-independent first SA."""
    first = decoder.layers[0]
    if not first.norm_first:
        raise ValueError("first-query self-attention cache requires norm-first decoder")
    hidden = cached_after_self_attention
    hidden = hidden + first._mha_block(first.norm2(hidden), memory, None, None)
    hidden = hidden + first._ff_block(first.norm3(hidden))
    for layer in decoder.layers[1:]:
        hidden = layer(hidden, memory)
    if decoder.norm is not None:
        hidden = decoder.norm(hidden)
    return hidden


def _make_first_self_attention_cache(uact: nn.Module, batch_size: int) -> torch.Tensor:
    if batch_size <= 0 or uact.training:
        raise ValueError("query cache requires a positive batch and eval-mode U-ACT")
    first = uact.action_decoder.layers[0]
    if not first.norm_first:
        raise ValueError("query cache requires norm-first decoder")
    queries = uact.action_queries.unsqueeze(0).expand(batch_size, -1, -1)
    with torch.inference_mode():
        return (queries + first._sa_block(first.norm1(queries), None, None)).detach()


class UACTInferenceCore(nn.Module):
    """Fixed-shape U-ACT forward without data-dependent validation branches."""

    def __init__(self, uact: nn.Module, *, cache_batch_size: int | None = None) -> None:
        super().__init__()
        self.uact = uact
        self.config = uact.config
        self.cache_batch_size = cache_batch_size
        if cache_batch_size is not None:
            self.register_buffer(
                "first_self_attention_cache",
                _make_first_self_attention_cache(uact, cache_batch_size),
            )

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        model = self.uact
        normalized_obs = (observations - model.obs_mean) / model.obs_std
        memory = model.obs_projection(normalized_obs)
        memory = model.observation_encoder(
            memory + model.observation_position.unsqueeze(0)
        )
        queries = model.action_queries.unsqueeze(0).expand(
            observations.shape[0], -1, -1
        )
        if self.cache_batch_size is None:
            hidden = model.action_decoder(queries, memory)
        else:
            if observations.shape[0] != self.cache_batch_size:
                raise ValueError("U-ACT query-cache batch size changed")
            hidden = _cached_first_decoder_layer(
                model.action_decoder, self.first_self_attention_cache, memory
            )
        normalized_mean = model.mean_head(hidden)
        logvar = model.logvar_head(hidden).clamp(
            model.config.min_logvar, model.config.max_logvar
        )
        mean = normalized_mean * model.action_std + model.action_mean
        continuous = logvar.exp()[..., : model.config.act_dim - 1]
        near_count = min(4, model.config.num_queries)
        risk_count = min(8, model.config.num_queries)
        near = continuous[:, :near_count].mean(dim=(1, 2))
        p95 = fixed_linear_quantile_95(continuous[:, :risk_count].flatten(start_dim=1))
        uncertainty = 0.7 * near + 0.3 * p95
        return (mean, uncertainty)


class UACTMeanInferenceCore(nn.Module):
    """Mean-only U-ACT path for steps where uncertainty cannot affect execution."""

    def __init__(self, uact: nn.Module, *, cache_batch_size: int | None = None) -> None:
        super().__init__()
        self.uact = uact
        self.config = uact.config
        self.cache_batch_size = cache_batch_size
        if cache_batch_size is not None:
            self.register_buffer(
                "first_self_attention_cache",
                _make_first_self_attention_cache(uact, cache_batch_size),
            )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        model = self.uact
        normalized_obs = (observations - model.obs_mean) / model.obs_std
        memory = model.obs_projection(normalized_obs)
        memory = model.observation_encoder(
            memory + model.observation_position.unsqueeze(0)
        )
        queries = model.action_queries.unsqueeze(0).expand(
            observations.shape[0], -1, -1
        )
        if self.cache_batch_size is None:
            hidden = model.action_decoder(queries, memory)
        else:
            if observations.shape[0] != self.cache_batch_size:
                raise ValueError("U-ACT query-cache batch size changed")
            hidden = _cached_first_decoder_layer(
                model.action_decoder, self.first_self_attention_cache, memory
            )
        normalized_mean = model.mean_head(hidden)
        return normalized_mean * model.action_std + model.action_mean


def fixed_linear_quantile_95(values: torch.Tensor) -> torch.Tensor:
    """Exact linear q=.95 quantile specialized for a fixed-size final axis.

    U-ACT routing always reduces a small, fixed vector.  Avoiding the generic
    quantile dispatcher preserves the same two order statistics and the same
    interpolation expression while cutting kernel-launch overhead.
    """
    if values.ndim != 2 or values.shape[1] <= 1 or values.dtype != torch.float32:
        raise ValueError("U-ACT fixed quantile expects float32 [batch, width>1]")
    rank = torch.tensor(0.95, dtype=torch.float32) * (values.shape[1] - 1)
    lower_index = int(torch.floor(rank))
    fraction = float(rank - lower_index)
    lower = torch.kthvalue(values, lower_index + 1, dim=1).values
    upper = torch.kthvalue(values, lower_index + 2, dim=1).values
    return torch.lerp(lower, upper, fraction)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def latency_summary(samples_ms: list[float]) -> dict[str, float]:
    if not samples_ms:
        raise ValueError("latency summary requires samples")
    return {
        "mean_ms": float(sum(samples_ms) / len(samples_ms)),
        "p50_ms": percentile(samples_ms, 0.5),
        "p95_ms": percentile(samples_ms, 0.95),
        "max_ms": max(samples_ms),
        "samples": len(samples_ms),
    }
