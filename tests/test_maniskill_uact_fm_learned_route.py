import torch
import torch.nn as nn
from src.policies.maniskill_uact_fm_learned_route import (
    LossDistributionHead,
    LossDistributionHeadConfig,
    PreOnlyRouterInferenceCore,
    ensemble_loss_distribution,
    risk_choice,
    uact_only_route_features,
)


def test_uact_features_are_flow_free_and_finite():
    observations = torch.randn(3, 2, 9)
    source = torch.randn(3, 8, 7)
    uncertainty = torch.tensor([0.1, 0.2, 0.3])
    features = uact_only_route_features(observations, source, uncertainty)
    assert features.shape == (3, 2 * 9 + 8 * 7 + 1)
    assert torch.isfinite(features).all()


def test_ensemble_combines_aleatoric_and_epistemic_uncertainty():
    config = LossDistributionHeadConfig(feature_dim=4, choices=3, hidden_dim=8)
    heads = nn.ModuleList([LossDistributionHead(config), LossDistributionHead(config)])
    for parameter in heads.parameters():
        nn.init.zeros_(parameter)
    mean, standard_deviation = ensemble_loss_distribution(
        heads,
        torch.zeros(2, 4),
        feature_mean=torch.zeros(4),
        feature_scale=torch.ones(4),
        target_location=torch.tensor(2.0),
        target_scale=torch.tensor(3.0),
    )
    torch.testing.assert_close(mean, torch.full((2, 3), 2.0))
    torch.testing.assert_close(standard_deviation, torch.full((2, 3), 3.0))


def test_risk_choice_has_no_implicit_c2_default():
    mean = torch.tensor([[0.1, 0.5, 0.6], [0.4, 0.2, 0.5], [0.9, 0.8, 0.1]])
    standard_deviation = torch.zeros_like(mean)
    costs = torch.tensor([1.0, 2.0, 3.0])
    assert risk_choice(
        mean, standard_deviation, costs, beta=0.0, compute_price=0.0, first_choice=1
    ).tolist() == [1, 2, 3]


def test_preonly_inference_core_matches_reference_ensemble_and_risk():
    torch.manual_seed(17)
    observations = torch.randn(5, 2, 9)
    source = torch.randn(5, 8, 7)
    uncertainty = torch.rand(5).clamp_min(0.01)
    feature_dim = 2 * 9 + 8 * 7 + 1
    config = LossDistributionHeadConfig(
        feature_dim=feature_dim, choices=4, hidden_dim=16
    )
    heads = nn.ModuleList([LossDistributionHead(config) for _ in range(3)]).eval()
    feature_mean = torch.randn(feature_dim)
    feature_scale = torch.rand(feature_dim).clamp_min(0.1)
    target_location = torch.tensor(0.7)
    target_scale = torch.tensor(1.3)
    costs = torch.tensor([0.0, 1 / 3, 2 / 3, 1.0])
    transition_bias = torch.rand(5, 4) * 0.2
    beta = 0.25
    compute_price = 0.1
    features = uact_only_route_features(observations, source, uncertainty)
    expected_mean, expected_std = ensemble_loss_distribution(
        heads,
        features,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        target_location=target_location,
        target_scale=target_scale,
    )
    expected_choice = risk_choice(
        expected_mean + transition_bias,
        expected_std,
        costs,
        beta=beta,
        compute_price=compute_price,
        first_choice=0,
    )
    core = PreOnlyRouterInferenceCore(
        heads,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        target_location=target_location,
        target_scale=target_scale,
        costs=costs,
        beta=beta,
        compute_price=compute_price,
    ).eval()
    actual_choice, actual_mean, actual_std = core(
        observations, source, uncertainty, transition_bias
    )
    torch.testing.assert_close(actual_mean, expected_mean, rtol=0, atol=0)
    torch.testing.assert_close(actual_std, expected_std, rtol=0, atol=0)
    torch.testing.assert_close(actual_choice, expected_choice, rtol=0, atol=0)
