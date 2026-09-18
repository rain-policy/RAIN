from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch
import mani_skill.envs
from scripts.rollout_utils import (
    atomic_json,
    current_info,
    scalar_bool,
    scalar_float,
    sha256,
    state_vector,
)
from scripts.load_refiner import EXECUTION_HORIZON, load_fixed_flow
from src.policies.maniskill_uact_current_action_flow import (
    aggregate_uact_chunk,
    temporal_aggregation_weight_table,
)
from src.policies.maniskill_uact_fm_learned_route import (
    LossDistributionHead,
    LossDistributionHeadConfig,
    PreOnlyRouterInferenceCore,
)
from src.policies.maniskill_uact_fresh_flow import load_frozen_uact
from src.policies.maniskill_uact_mars_action_flow import (
    build_causal_source_chunk,
    fill_causal_source_chunk,
)
from src.runtime.maniskill_fair_inference import (
    DeviceTimer,
    DualStreamUACTRunner,
    FixedStepRefiner,
    ObservationBuffer,
    UACTInferenceCore,
    UACTMeanInferenceCore,
    add_fair_runtime_args,
    autocast_context,
    backfill_temporal_aggregation_proposals,
    compile_module,
    configure_torch_runtime,
    graph_module,
    latency_summary,
    runtime_from_args,
)


def load_learned_heads(state: dict, device: torch.device) -> dict:
    if state.get("method") != "peg_uact_fm_fully_learned_heads_train_v1":
        raise RuntimeError("invalid fully learned router checkpoint")
    result = {}
    for name in ("uact_head", "fm_head"):
        values = state[name]
        config = LossDistributionHeadConfig(**values["config"])
        heads = torch.nn.ModuleList(
            [LossDistributionHead(config) for _ in values["ensemble"]]
        ).to(device)
        for head, member in zip(heads, values["ensemble"]):
            head.load_state_dict(member, strict=True)
            head.eval()
            for parameter in head.parameters():
                parameter.requires_grad_(False)
        result[name] = heads
        result[f"{name}_feature_mean"] = values["feature_mean"].to(device)
        result[f"{name}_feature_scale"] = values["feature_scale"].to(device)
    result["target_location"] = state["target_location"].to(device)
    result["target_scale"] = state["target_scale"].to(device)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--render-backend",
        default="gpu",
        help="ManiSkill rendering backend: gpu, none, or a PCI device",
    )
    parser.add_argument("--uact-checkpoint", required=True)
    parser.add_argument("--flow-checkpoint", required=True)
    parser.add_argument("--route", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed-start", type=int, default=50000)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--env-id", default="PegInsertionSide-v1")
    parser.add_argument("--control-mode", default="pd_ee_delta_pose")
    parser.add_argument("--max-episode-steps", type=int, default=320)
    parser.add_argument("--aggregation-k", type=float, default=0.01)
    parser.add_argument(
        "--parallel-uact-streams",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="overlap deferred mean-only ACT with the current ACT decision on CUDA",
    )
    parser.add_argument("--cache-first-query-self-attention", action="store_true")
    parser.add_argument("--posthoc-diagnostic", action="store_true")
    parser.add_argument(
        "--capture-episode-media",
        action="store_true",
        help="Record aligned RGB frames and per-step ACT-uncertainty/C traces.",
    )
    parser.add_argument(
        "--capture-episode-trace",
        action="store_true",
        help="Record per-step ACT-uncertainty/C traces without rendering video.",
    )
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    add_fair_runtime_args(parser)
    return parser.parse_args()


@torch.inference_mode()
def rollout(
    args,
    decision_uact,
    decision_uact_batch2,
    deferred_uact,
    flow,
    head,
    route,
    env,
    runtime,
    ta_weights,
    observation_buffer,
    observation_buffer2,
    deferred_buffer,
    preonly_runtime,
    dual_stream_uact,
    *,
    seed: int,
) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    observation, info = env.reset(seed=seed)
    state = state_vector(observation)
    history = [state.copy(), state.copy()]
    horizon = decision_uact.config.num_queries
    storage = torch.zeros(
        args.max_episode_steps,
        args.max_episode_steps + horizon,
        flow.config.act_dim,
        device=args.device_object,
    )
    cached_chunk = None
    cached_horizon = 0
    cached_c = 0
    previous_route_c = None
    previous_route_horizon = None
    block_remaining = 0
    deferred_histories: list[np.ndarray] = []
    block_histogram = [0, 0, 0, 0]
    step_histogram = [0, 0, 0, 0]
    gain1_sum = gain2_sum = logvar_sum = 0.0
    learned_uact_margin_sum = learned_fm_std_sum = 0.0
    flow_nfe = boundary_count = fm_boundary_count = 0
    policy_latency_samples = []
    boundary_latency_samples = []
    episode_trace = []
    frames = []
    active_uncertainty = None
    success_once = scalar_bool(info.get("success"))
    final_success = success_once
    return_sum = 0.0
    episode_length = 0
    for timestep in range(args.max_episode_steps):
        boundary = block_remaining == 0
        diagnostic_uact_margin = None
        diagnostic_fm_std = None
        diagnostic_prediction = None
        host_history = np.stack(history)
        host_observations = torch.from_numpy(host_history[None])
        timer = DeviceTimer(args.device_object)
        if runtime.include_input_transfer:
            timer.start()
        base = None
        if boundary:
            observations = observation_buffer.update(host_observations)
            if deferred_histories:
                missing = len(deferred_histories)
                if missing == EXECUTION_HORIZON - 1:
                    deferred_observations = deferred_buffer.update(
                        torch.from_numpy(np.stack(deferred_histories))
                    )
                    if not runtime.include_input_transfer:
                        timer.start()
                    if dual_stream_uact is None:
                        with autocast_context(runtime, args.device_object):
                            proposal_batch = deferred_uact(deferred_observations)
                            proposal_mean, uncertainty = decision_uact(observations)
                    else:
                        proposal_batch, proposal_mean, uncertainty = (
                            dual_stream_uact.run(
                                deferred_uact,
                                decision_uact,
                                deferred_observations,
                                observations,
                                runtime,
                            )
                        )
                    backfill_temporal_aggregation_proposals(
                        storage,
                        proposal_batch.to(torch.float32),
                        first_timestep=timestep - missing,
                        proposal_horizon=horizon,
                    )
                else:
                    raise RuntimeError(
                        f"unsupported deferred observation count: {missing}"
                    )
                deferred_histories.clear()
            else:
                if not runtime.include_input_transfer:
                    timer.start()
                with autocast_context(runtime, args.device_object):
                    proposal_mean, uncertainty = decision_uact(observations)
            proposal_mean = proposal_mean.to(torch.float32)
            uncertainty = uncertainty.to(torch.float32)
            base = aggregate_uact_chunk(
                storage,
                proposal_mean[0],
                timestep=timestep,
                aggregation_k=args.aggregation_k,
                weight_table=ta_weights,
            ).clamp(-1, 1)
            route_method = "peg_uact_fm_preonly_route_v1"
            uncertainty_route = False
            learned_route = True
            preonly_fast = True
            source = None
            selected_preonly = None
            transition_energy = None
            source = preonly_runtime["source_buffer"]
            fill_causal_source_chunk(source, base[None], proposal_mean)
            if previous_route_c is None:
                transition_bias = preonly_runtime["zero_transition_bias"]
            else:
                transition_bias = preonly_runtime["transition_bias_by_horizon"][
                    previous_route_horizon
                ][previous_route_c : previous_route_c + 1]
            with autocast_context(runtime, args.device_object):
                uact_choice, uact_mean, uact_std = preonly_runtime["router"](
                    observations, source, uncertainty, transition_bias
                )
            selected_preonly = int(uact_choice.item())
            use_c0_fast = selected_preonly == 0
            diagnostic_uact_margin = uact_mean[:, 1:].amin(dim=1) - uact_mean[:, 0]
            if use_c0_fast:
                selected_c = 0
                cached_chunk = source[0, :8]
                cached_horizon = 8
            else:
                if source is None:
                    source = build_causal_source_chunk(
                        base[None],
                        proposal_mean,
                        chunk_horizon=flow.config.chunk_horizon,
                    )
                selected_c = selected_preonly
                if selected_c not in preonly_runtime["refiners"]:
                    raise RuntimeError("pre-only router selected an invalid positive C")
                with autocast_context(runtime, args.device_object):
                    refined = preonly_runtime["refiners"][selected_c](
                        observations, source
                    )
                cached_chunk = refined[0]
                cached_horizon = EXECUTION_HORIZON
                fm_boundary_count += 1
            cached_c = selected_c
            block_remaining = cached_horizon
            previous_route_c = selected_c
            previous_route_horizon = cached_horizon
            block_histogram[selected_c] += 1
            flow_nfe += selected_c
            boundary_count += 1
        else:
            if not runtime.include_input_transfer:
                timer.start()
            deferred_histories.append(host_history.copy())
        block_index = cached_horizon - block_remaining
        executed_c = cached_c
        action = cached_chunk[block_index]
        step_histogram[executed_c] += 1
        block_remaining -= 1
        if block_remaining == 0:
            cached_chunk = None
            cached_c = 0
            cached_horizon = 0
        elapsed = timer.stop_ms()
        policy_latency_samples.append(elapsed)
        if boundary:
            boundary_latency_samples.append(elapsed)
            if args.capture_episode_media or args.capture_episode_trace:
                active_uncertainty = float(uncertainty.item())
            if diagnostic_uact_margin is not None:
                learned_uact_margin_sum += float(diagnostic_uact_margin.item())
            if diagnostic_fm_std is not None:
                learned_fm_std_sum += float(diagnostic_fm_std.item())
            if diagnostic_prediction is not None:
                gain1_sum += float(diagnostic_prediction["gain_after_c1"].item())
                gain2_sum += float(diagnostic_prediction["gain_after_c2"].item())
                logvar_sum += float(diagnostic_prediction["logvar_frontloaded"].item())
        observation, reward, terminated, truncated, info = env.step(
            action.clamp(-1, 1).cpu().numpy()
        )
        episode_length = timestep + 1
        return_sum += scalar_float(reward)
        state = state_vector(observation)
        history = [history[-1], state.copy()]
        episode_info = current_info(info)
        current_success = scalar_bool(episode_info.get("success"))
        success_once = success_once or current_success
        final_success = current_success
        if args.capture_episode_media or args.capture_episode_trace:
            if active_uncertainty is None:
                raise RuntimeError(
                    "ACT uncertainty was unavailable for a captured step"
                )
            episode_trace.append(
                {
                    "step": timestep,
                    "decision_boundary": boundary,
                    "act_uncertainty": active_uncertainty,
                    "selected_c": executed_c,
                    "success": current_success,
                }
            )
            if args.capture_episode_media:
                rendered = env.render()
                if torch.is_tensor(rendered):
                    frame = rendered.detach().cpu().numpy().copy()
                else:
                    frame = np.asarray(rendered).copy()
                if frame.ndim == 4 and frame.shape[0] == 1:
                    frame = frame[0]
                if frame.ndim != 3 or frame.shape[-1] not in (3, 4):
                    raise RuntimeError(
                        f"render returned an invalid image shape: {frame.shape}"
                    )
                frames.append(frame[..., :3])
        if scalar_bool(truncated):
            break
    result = {
        "seed": seed,
        "success_once": success_once,
        "success_at_end": final_success,
        "return": return_sum,
        "episode_len": episode_length,
        "c_block_histogram": block_histogram,
        "c_environment_step_histogram": step_histogram,
        "flow_nfe": flow_nfe,
        "mean_c_per_block": flow_nfe / boundary_count,
        "fm_boundary_count": fm_boundary_count,
        "mean_predicted_gain_after_c1": gain1_sum / max(fm_boundary_count, 1),
        "mean_predicted_gain_after_c2": gain2_sum / max(fm_boundary_count, 1),
        "mean_predicted_frontloaded_logvar": logvar_sum / max(fm_boundary_count, 1),
        "mean_learned_uact_enter_margin": learned_uact_margin_sum
        / max(boundary_count, 1),
        "mean_learned_fm_predictive_std": learned_fm_std_sum
        / max(fm_boundary_count, 1),
        "policy_latency_samples_ms": policy_latency_samples,
        "boundary_latency_samples_ms": boundary_latency_samples,
    }
    if args.capture_episode_media or args.capture_episode_trace:
        result["episode_trace"] = episode_trace
    if args.capture_episode_media:
        result["_frames"] = frames
    return result


def main() -> None:
    args = parse_args()
    from mani_skill.envs.utils.system import backend as backend_module

    original_parse = backend_module.parse_backend_device_id
    backend_module.parse_backend_device_id = lambda value: (
        (value, None)
        if isinstance(value, str) and value.startswith("pci:")
        else original_parse(value)
    )
    if min(args.episodes, args.max_episode_steps) <= 0:
        raise ValueError("evaluation counts must be positive")
    if args.video_fps <= 0:
        raise ValueError("video FPS must be positive")
    args.device_object = torch.device(
        "cuda"
        if args.device == "cuda"
        or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    runtime = runtime_from_args(args, args.device_object)
    if args.parallel_uact_streams and args.device_object.type != "cuda":
        raise ValueError("parallel U-ACT streams require CUDA")
    configure_torch_runtime(runtime, args.device_object)
    output = Path(args.out).expanduser().resolve()
    uact_path = Path(args.uact_checkpoint).expanduser().resolve()
    flow_path = Path(args.flow_checkpoint).expanduser().resolve()
    route_path = Path(args.route).expanduser().resolve()
    route = json.loads(route_path.read_text(encoding="utf-8"))
    if route.get("method") != "peg_uact_fm_preonly_route_v1":
        raise ValueError("This release supports only the RAIN route")
    route_method = "peg_uact_fm_preonly_route_v1"
    benefit_method = False
    incremental_method = False
    cumulative_method = False
    risk_method = True
    direct_method = False
    uncertainty_method = False
    learned_method = True
    preonly_fast = True
    if route["flow_checkpoint"]["sha256"] != sha256(flow_path):
        raise RuntimeError("route and Flow checkpoint hashes disagree")
    head_identity = route["router_checkpoint"]
    head_path = Path(head_identity["path"]).expanduser().resolve()
    if head_identity["sha256"] != sha256(head_path):
        raise RuntimeError("route and router-head hashes disagree")
    head_state = torch.load(head_path, map_location="cpu", weights_only=False)
    head = load_learned_heads(head_state, args.device_object)
    head["costs"] = torch.as_tensor(
        route["costs"], device=args.device_object, dtype=torch.float32
    )
    if "transition_probability" in route:
        head["transition_probability"] = torch.as_tensor(
            route["transition_probability"],
            device=args.device_object,
            dtype=torch.float32,
        )
    flow, flow_state = load_fixed_flow(flow_path, args.device_object)
    uact, provenance = load_frozen_uact(uact_path, device=args.device_object)
    if provenance["sha256"] != flow_state["uact_source"]["sha256"]:
        raise RuntimeError("Flow and U-ACT hashes disagree")
    decision_uact = compile_module(
        UACTInferenceCore(
            uact, cache_batch_size=1 if args.cache_first_query_self_attention else None
        ).eval(),
        runtime,
    )
    decision_uact_batch2 = None
    deferred_uact = compile_module(
        UACTMeanInferenceCore(
            uact,
            cache_batch_size=(
                EXECUTION_HORIZON - 1 if args.cache_first_query_self_attention else None
            ),
        ).eval(),
        runtime,
    )
    dual_stream_uact = (
        DualStreamUACTRunner(args.device_object) if args.parallel_uact_streams else None
    )
    ta_weights = temporal_aggregation_weight_table(
        provenance["model_config"]["num_queries"],
        args.aggregation_k,
        device=args.device_object,
    )
    observation_buffer = ObservationBuffer(
        (1, 2, flow.config.obs_dim), args.device_object
    )
    observation_buffer2 = None
    deferred_buffer = ObservationBuffer(
        (EXECUTION_HORIZON - 1, 2, flow.config.obs_dim), args.device_object
    )
    preonly_runtime = None
    source_buffer = torch.empty(
        1,
        flow.config.chunk_horizon,
        flow.config.act_dim,
        device=args.device_object,
        dtype=torch.float32,
    )
    zero_transition_bias = torch.zeros(
        1, 4, device=args.device_object, dtype=torch.float32
    )
    transition_bias_by_horizon = {
        8: torch.zeros(4, 4, device=args.device_object, dtype=torch.float32),
        EXECUTION_HORIZON: torch.zeros(
            4, 4, device=args.device_object, dtype=torch.float32
        ),
    }
    if "transition_probability" in head:
        strength = float(route["transition_strength"])
        transition_bias_by_horizon = {
            value: -torch.log(
                torch.linalg.matrix_power(
                    head["transition_probability"], value
                ).clamp_min(1e-09)
            )
            * strength
            for value in {8, EXECUTION_HORIZON}
        }
    router_module = PreOnlyRouterInferenceCore(
        head["uact_head"],
        feature_mean=head["uact_head_feature_mean"],
        feature_scale=head["uact_head_feature_scale"],
        target_location=head["target_location"],
        target_scale=head["target_scale"],
        costs=head["costs"],
        beta=float(route["beta"]),
        compute_price=float(route["compute_price"]),
    ).eval()
    router_core = compile_module(router_module, runtime)
    refiners = {
        value: compile_module(FixedStepRefiner(flow, value).eval(), runtime)
        for value in (1, 2, 3)
    }
    preonly_runtime = {
        "source_buffer": source_buffer,
        "zero_transition_bias": zero_transition_bias,
        "transition_bias_by_horizon": transition_bias_by_horizon,
        "router": router_core,
        "refiners": refiners,
    }
    dummy = observation_buffer.tensor.zero_()
    decision_uact = graph_module(decision_uact, runtime, (dummy,))
    deferred_uact = graph_module(
        deferred_uact, runtime, (deferred_buffer.tensor.zero_(),)
    )
    preonly_runtime["router"] = graph_module(
        preonly_runtime["router"],
        runtime,
        (
            dummy,
            preonly_runtime["source_buffer"],
            torch.zeros(1, device=args.device_object, dtype=torch.float32),
            preonly_runtime["zero_transition_bias"],
        ),
    )
    preonly_runtime["refiners"] = {
        value: graph_module(module, runtime, (dummy, preonly_runtime["source_buffer"]))
        for (value, module) in preonly_runtime["refiners"].items()
    }
    with torch.inference_mode():
        for _ in range(runtime.warmup_calls):
            with autocast_context(runtime, args.device_object):
                warm_mean, warm_uncertainty = decision_uact(dummy)
                if dual_stream_uact is None:
                    deferred_uact(deferred_buffer.tensor.zero_())
                else:
                    dual_stream_uact.run(
                        deferred_uact,
                        decision_uact,
                        deferred_buffer.tensor.zero_(),
                        dummy,
                        runtime,
                    )
            source = build_causal_source_chunk(
                warm_mean[:, 0].to(torch.float32),
                warm_mean.to(torch.float32),
                chunk_horizon=flow.config.chunk_horizon,
            )
            preonly_runtime["source_buffer"].copy_(source)
            preonly_runtime["router"](
                dummy,
                preonly_runtime["source_buffer"],
                warm_uncertainty.to(torch.float32),
                preonly_runtime["zero_transition_bias"],
            )
            for refiner in preonly_runtime["refiners"].values():
                refiner(dummy, preonly_runtime["source_buffer"])
    if args.device_object.type == "cuda":
        torch.cuda.synchronize(args.device_object)
    contract = {
        "schema_version": 2,
        "method": "peg_uact_fm_fully_learned_route_eval_v1",
        "posthoc_diagnostic": args.posthoc_diagnostic,
        "uact_checkpoint": str(uact_path),
        "uact_sha256": sha256(uact_path),
        "flow_checkpoint": str(flow_path),
        "flow_sha256": sha256(flow_path),
        "head_checkpoint": str(head_path),
        "head_sha256": sha256(head_path),
        "route": str(route_path),
        "route_sha256": sha256(route_path),
        "threshold_after_c1": route.get("threshold_after_c1"),
        "threshold_after_c2": route.get("threshold_after_c2"),
        "beta": route.get("beta"),
        "gamma": route.get("gamma"),
        "compute_price": route.get("compute_price"),
        "transition_strength": route.get("transition_strength"),
        "transition_probability": route.get("transition_probability"),
        "seed_start": args.seed_start,
        "episodes": args.episodes,
        "max_episode_steps": args.max_episode_steps,
        "execution_horizon": f"adaptive_{8}_or_8",
        "c0_fast": {
            "enabled": True,
            "decision": "fully_learned_no_fixed_fraction",
            "execution_horizon": 8,
        },
        "uncertainty_ceiling": route.get("uncertainty_ceiling"),
        "termination_handling": "ignore_task_termination_stop_on_time_limit",
        "solver": "original_uniform_euler_fixed_step_cuda_graph",
        "runtime": runtime.to_dict(),
        "preonly_fast_path": {
            "enabled": True,
            "router_cuda_graph": runtime.backend == "cudagraph",
            "refiner_cuda_graphs": runtime.backend == "cudagraph",
            "positive_choices": [1, 2, 3],
            "online_post_fm_features": False,
            "transition_matrix_powers_precomputed": "transition_probability" in head,
            "choice_device_to_host_syncs_per_boundary": 1,
            "parallel_uact_streams": args.parallel_uact_streams,
            "inplace_source_chunk": True,
        },
        "capture_episode_media": args.capture_episode_media,
        "capture_episode_trace": args.capture_episode_trace,
        "video_fps": args.video_fps if args.capture_episode_media else None,
    }
    contract_path = output / "contract.json"
    progress_path = output / "progress.jsonl"
    if output.exists():
        if (
            not contract_path.is_file()
            or json.loads(contract_path.read_text()) != contract
        ):
            raise RuntimeError("evaluation resume contract changed")
        results = (
            [
                json.loads(line)
                for line in progress_path.read_text().splitlines()
                if line.strip()
            ]
            if progress_path.is_file()
            else []
        )
    else:
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(contract, contract_path)
        results = []
    completed = {int(item["seed"]) for item in results}
    if len(completed) != len(results):
        raise RuntimeError("evaluation progress contains duplicate seeds")
    env = gym.make(
        args.env_id,
        control_mode=args.control_mode,
        reward_mode="sparse",
        obs_mode="state",
        render_mode="rgb_array" if args.capture_episode_media else None,
        max_episode_steps=args.max_episode_steps,
        reconfiguration_freq=1,
        render_backend=args.render_backend,
    )
    try:
        for seed in range(args.seed_start, args.seed_start + args.episodes):
            if seed in completed:
                continue
            result = rollout(
                args,
                decision_uact,
                decision_uact_batch2,
                deferred_uact,
                flow,
                head,
                route,
                env,
                runtime,
                ta_weights,
                observation_buffer,
                observation_buffer2,
                deferred_buffer,
                preonly_runtime,
                dual_stream_uact,
                seed=seed,
            )
            frames = result.pop("_frames", None)
            if frames is not None:
                from mani_skill.utils.visualization.misc import images_to_video

                video_dir = output / "videos"
                video_dir.mkdir(parents=True, exist_ok=True)
                video_stem = f"seed_{seed}_{('success' if result['success_once'] else 'failure')}"
                images_to_video(
                    frames,
                    output_dir=str(video_dir),
                    video_name=video_stem,
                    fps=args.video_fps,
                    quality=8,
                    verbose=False,
                )
                result["video_path"] = str(video_dir / f"{video_stem}.mp4")
                result["video_frames"] = len(frames)
            results.append(result)
            with progress_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, sort_keys=True) + "\n")
            print(json.dumps(result, sort_keys=True), flush=True)
    finally:
        env.close()
    histogram = [
        sum((item["c_block_histogram"][c] for item in results)) for c in range(4)
    ]
    total_blocks = sum(histogram)
    policy_latency = [
        value for item in results for value in item["policy_latency_samples_ms"]
    ]
    boundary_latency = [
        value for item in results for value in item["boundary_latency_samples_ms"]
    ]
    payload = {
        **contract,
        "actual_episodes": len(results),
        "success_once_count": sum((item["success_once"] for item in results)),
        "success_at_end_count": sum((item["success_at_end"] for item in results)),
        "c_block_histogram": histogram,
        "mean_c_per_block": sum((item["flow_nfe"] for item in results)) / total_blocks,
        "policy_latency": latency_summary(policy_latency),
        "boundary_latency": latency_summary(boundary_latency),
        "episodes_detail": results,
    }
    atomic_json(payload, output / "results.json")
    print(
        json.dumps(
            {
                key: value
                for (key, value) in payload.items()
                if key != "episodes_detail"
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
