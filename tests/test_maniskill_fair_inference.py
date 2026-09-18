import torch
from src.policies.maniskill_uact_current_action_flow import (
    aggregate_uact_chunk,
    temporal_aggregation_weight_table,
)
from src.policies.maniskill_uact_mars_action_flow import (
    MARSActionFlowConfig,
    UACTMARSActionFlow,
    build_causal_source_chunk,
    fill_causal_source_chunk,
)
from src.policies.maniskill_uncertain_act import CleanUncertainACT, UncertainACTConfig
from src.runtime.maniskill_fair_inference import (
    FairRuntimeConfig,
    CUDAGraphInferenceWrapper,
    DualStreamUACTRunner,
    FixedStepRefiner,
    UACTInferenceCore,
    UACTMeanInferenceCore,
    compile_module,
    fixed_linear_quantile_95,
    latency_summary,
)


def test_fixed_linear_quantile_95_is_bitwise_equal_to_torch_quantile():
    torch.manual_seed(3)
    cases = [torch.randn(1, width) for width in (24, 40, 48)]
    cases.extend(
        [
            torch.randn(17, 48),
            torch.zeros(4, 48),
            torch.arange(48, dtype=torch.float32).repeat(3, 1),
        ]
    )
    for values in cases:
        expected = torch.quantile(values, 0.95, dim=1)
        actual = fixed_linear_quantile_95(values)
        assert torch.equal(actual, expected)


def test_allocation_free_causal_source_is_bitwise_equal():
    torch.manual_seed(5)
    current = torch.randn(3, 7)
    proposal = torch.randn(3, 30, 7)
    expected = build_causal_source_chunk(current, proposal, chunk_horizon=8)
    output = torch.empty_like(expected)
    actual = fill_causal_source_chunk(output, current, proposal)
    assert actual.data_ptr() == output.data_ptr()
    assert torch.equal(actual, expected)


def test_dual_stream_runner_rejects_cpu():
    try:
        DualStreamUACTRunner(torch.device("cpu"))
    except ValueError as error:
        assert "requires CUDA" in str(error)
    else:
        raise AssertionError("CPU dual-stream construction should fail closed")


def test_cuda_graph_wrapper_rejects_cpu_inputs():
    try:
        CUDAGraphInferenceWrapper(torch.nn.Identity(), (torch.zeros(1),))
    except ValueError as error:
        assert "requires" in str(error)
    else:
        raise AssertionError("CPU CUDA Graph capture should fail closed")


def test_default_compile_backend_explicitly_disables_cuda_graphs(monkeypatch):
    calls = []

    def fake_compile(module, **kwargs):
        calls.append(kwargs)
        return module

    monkeypatch.setattr(torch, "compile", fake_compile)
    module = torch.nn.Identity()
    config = FairRuntimeConfig(backend="compile", compile_mode="default")
    assert compile_module(module, config) is module
    assert calls == [
        {"fullgraph": True, "dynamic": False, "options": {"triton.cudagraphs": False}}
    ]


def make_runtime_flow() -> UACTMARSActionFlow:
    return UACTMARSActionFlow(
        MARSActionFlowConfig(
            obs_dim=3,
            act_dim=7,
            arm_dim=6,
            obs_horizon=2,
            chunk_horizon=8,
            hidden_dim=32,
            depth=2,
            num_heads=4,
            mlp_ratio=2,
            max_steps=2,
        ),
        obs_mean=torch.zeros(3),
        obs_std=torch.ones(3),
        action_mean=torch.zeros(7),
        action_std=torch.ones(7),
    )


def test_latency_summary_reports_distribution_not_only_mean():
    summary = latency_summary([1.0, 2.0, 3.0, 10.0])
    assert summary["mean_ms"] == 4.0
    assert summary["p50_ms"] == 2.5
    assert summary["p95_ms"] > 8.0
    assert summary["max_ms"] == 10.0
    assert summary["samples"] == 4


def test_precomputed_ta_weights_match_original_formula_exactly():
    torch.manual_seed(9)
    chunks = torch.randn(6, 5, 3)
    legacy_storage = torch.zeros(6, 11, 3)
    cached_storage = torch.zeros_like(legacy_storage)
    table = temporal_aggregation_weight_table(5, 0.01, device=torch.device("cpu"))
    for timestep, chunk in enumerate(chunks):
        legacy = aggregate_uact_chunk(
            legacy_storage, chunk, timestep=timestep, aggregation_k=0.01
        )
        cached = aggregate_uact_chunk(
            cached_storage,
            chunk,
            timestep=timestep,
            aggregation_k=0.01,
            weight_table=table,
        )
        assert torch.equal(legacy, cached)


def test_fixed_refiner_runtime_core_matches_checked_policy_path():
    torch.manual_seed(11)
    model = make_runtime_flow().eval()
    with torch.no_grad():
        model.velocity_net.output.weight.normal_(std=0.01)
        model.velocity_net.final_modulation[-1].weight.normal_(std=0.01)
        for block in model.velocity_net.blocks:
            block.ada_ln[-1].weight.normal_(std=0.01)
    observations = torch.randn(2, 2, 3)
    source = torch.randn(2, 8, 7)
    for steps in (1, 2):
        expected = model.refine_chunk(observations, source, steps=steps)
        actual = FixedStepRefiner(model, steps)(observations, source)
        assert torch.equal(actual, expected)


def test_uact_runtime_core_matches_checked_policy_path():
    torch.manual_seed(13)
    model = CleanUncertainACT(
        UncertainACTConfig(
            obs_dim=3,
            act_dim=7,
            obs_horizon=2,
            num_queries=10,
            d_model=32,
            nhead=4,
            encoder_layers=1,
            decoder_layers=1,
            dim_feedforward=64,
            dropout=0.0,
        )
    ).eval()
    observations = torch.randn(4, 2, 3)
    expected = model(observations)
    mean, uncertainty = UACTInferenceCore(model)(observations)
    assert torch.equal(mean, expected.mean)
    assert torch.equal(uncertainty, expected.uncertainty)
    assert torch.equal(UACTMeanInferenceCore(model)(observations), expected.mean)
    cached_core = UACTInferenceCore(model, cache_batch_size=4)
    cached_mean_core = UACTMeanInferenceCore(model, cache_batch_size=4)
    with torch.inference_mode():
        inference_expected = model(observations)
        cached_mean, cached_uncertainty = cached_core(observations)
        assert torch.equal(cached_mean, inference_expected.mean)
        assert torch.equal(cached_uncertainty, inference_expected.uncertainty)
        assert torch.equal(cached_mean_core(observations), inference_expected.mean)
