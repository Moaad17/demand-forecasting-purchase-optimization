"""
Tests pour la Phase 3 (optimisation des stocks sous incertitude) :
génération de scénarios (constraints.py -- 3 scénarios directement
depuis les quantiles du forecasting), PDE direct (model.py / solver.py),
métriques WS/HN/EV/EVPI/VSS (solver.py), et décomposition L-shaped
(lshaped.py), utilisée directement pour résoudre HN.
"""

import numpy as np
import pandas as pd
import pytest

from optimization.constraints import generate_demand_scenarios
from optimization.solver import (
    solve_recourse_problem,
    solve_optimal,
    evaluate_strategies
)
from optimization.lshaped import solve_lshaped
from optimization.baselines import compute_unit_costs


UNIT_COST = 10
HOLDING_COST = 5
SHORTAGE_COST = 20


def _fake_forecast_result(mean: float, half_width: float) -> dict:
    """
    Résultat de forecast minimal à horizon=1 (predictions/lower/upper
    à un seul élément), pour tester l'optimisation indépendamment du
    module de forecasting (Phase 2). `half_width` est directement la
    demi-largeur de l'intervalle [lower, upper] -- aucune conversion
    statistique, generate_demand_scenarios utilise ces valeurs telles
    quelles comme scénarios.
    """

    return {
        "predictions": [mean],
        "lower": [mean - half_width],
        "upper": [mean + half_width],
        "dates": None,
        "mae": None
    }


def _optimize(forecast_results, integer=True, **kwargs):
    """Raccourci de test : génère les scénarios (3 quantiles) puis
    résout avec solve_optimal."""

    scenarios, probabilities = generate_demand_scenarios(forecast_results)
    return solve_optimal(scenarios, probabilities, integer=integer, **kwargs)


# ==========================================================
# generate_demand_scenarios (3 quantiles directs)
# ==========================================================

def test_generate_demand_scenarios_probabilities_are_fixed():
    forecast_results = {"A": _fake_forecast_result(mean=100, half_width=20)}

    _, probabilities = generate_demand_scenarios(forecast_results)

    assert list(probabilities) == pytest.approx([0.05, 0.90, 0.05])
    assert probabilities.sum() == pytest.approx(1.0)


def test_generate_demand_scenarios_uses_quantiles_directly():
    """Les 3 scénarios doivent être EXACTEMENT lower, predictions, upper
    -- aucune ré-estimation statistique."""

    forecast_results = {"A": _fake_forecast_result(mean=100, half_width=20)}

    scenarios, _ = generate_demand_scenarios(forecast_results)

    assert list(scenarios["A"]) == pytest.approx([80, 100, 120])


def test_generate_demand_scenarios_clips_negative_lower_bound():
    forecast_results = {"A": _fake_forecast_result(mean=10, half_width=50)}  # lower < 0

    scenarios, _ = generate_demand_scenarios(forecast_results)

    assert (scenarios["A"] >= 0).all()
    assert scenarios["A"][0] == 0  # lower clippé à 0, pas -40


def test_generate_demand_scenarios_rejects_multi_month_horizon():
    """Le pipeline ne raisonne que mois par mois (horizon=1) -- un
    résultat à plusieurs mois doit être rejeté explicitement plutôt que
    silencieusement tronqué."""

    forecast_results = {
        "A": {"predictions": [100, 110, 120], "lower": [90, 95, 100], "upper": [110, 120, 130]}
    }

    with pytest.raises(ValueError, match="horizon"):
        generate_demand_scenarios(forecast_results)


def test_generate_demand_scenarios_correlated_across_products():
    """Les scénarios sont corrélés entre produits : l'indice 0 = tous
    les produits à leur borne basse simultanément (même indice de
    scénario pour tous)."""

    forecast_results = {
        "A": _fake_forecast_result(mean=100, half_width=20),
        "B": _fake_forecast_result(mean=50, half_width=10),
    }

    scenarios, _ = generate_demand_scenarios(forecast_results)

    assert scenarios["A"][0] == pytest.approx(80)   # A, pessimiste
    assert scenarios["B"][0] == pytest.approx(40)   # B, pessimiste (même indice)


# ==========================================================
# solve_recourse_problem / solve_optimal (PDE direct, HN)
# ==========================================================

def test_optimize_respects_budget_and_capacity():
    forecast_results = {"Produit": _fake_forecast_result(mean=1000, half_width=100)}

    result = _optimize(
        forecast_results,
        unit_cost=UNIT_COST, holding_cost=HOLDING_COST, shortage_cost=SHORTAGE_COST,
        max_budget=2000, max_capacity=10**9
    )

    assert result["status"] == "Optimal"
    assert result["orders"]["Produit"] * UNIT_COST <= 2000 + 1e-6


def test_optimize_orders_are_integers_by_default():
    forecast_results = {
        "A": _fake_forecast_result(mean=137, half_width=15),
        "B": _fake_forecast_result(mean=283, half_width=30),
    }

    result = _optimize(
        forecast_results,
        unit_cost=UNIT_COST, holding_cost=HOLDING_COST, shortage_cost=SHORTAGE_COST,
        max_budget=10**9, max_capacity=10**9
    )

    for order in result["orders"].values():
        assert order == pytest.approx(round(order))


def test_optimize_infeasible_reports_status_without_crashing():
    forecast_results = {"Produit": _fake_forecast_result(mean=100, half_width=10)}

    result = _optimize(
        forecast_results,
        unit_cost=UNIT_COST, holding_cost=HOLDING_COST, shortage_cost=SHORTAGE_COST,
        max_budget=-1, max_capacity=10**9
    )

    assert result["status"] != "Optimal"
    assert result["orders"] == {}


# ==========================================================
# Décision à 3 points : la commande optimale ne peut être QUE
# lower, predictions ou upper (discrétisation minimale assumée)
# ==========================================================

def test_order_snaps_to_median_for_moderate_critical_ratio():
    """
    CR = (shortage_cost - unit_cost) / (shortage_cost + holding_cost).
    Avec UNIT_COST=10, HOLDING=5, SHORTAGE=20 -> CR=0.4, qui tombe dans
    la plage [0.05, 0.95] couverte par le scénario médian (proba 0.90)
    -> la commande optimale doit être EXACTEMENT la prédiction
    ponctuelle (aucune valeur intermédiaire n'existe avec seulement 3
    scénarios).
    """

    forecast_results = {"Produit": _fake_forecast_result(mean=1000, half_width=150)}

    result = _optimize(
        forecast_results,
        unit_cost=UNIT_COST, holding_cost=HOLDING_COST, shortage_cost=SHORTAGE_COST,
        max_budget=10**9, max_capacity=10**9, integer=False
    )

    assert result["orders"]["Produit"] == pytest.approx(1000, abs=1e-3)


def test_order_snaps_to_lower_for_low_critical_ratio():
    """CR très faible (rupture presque gratuite comparée au stockage)
    -> mieux vaut sous-commander -> la commande optimale tombe sur le
    scénario pessimiste (lower)."""

    unit_cost, holding_cost, shortage_cost = 10, 20, 11  # CR = (11-10)/(11+20) ≈ 0.032 < 0.05

    forecast_results = {"Produit": _fake_forecast_result(mean=1000, half_width=150)}

    result = _optimize(
        forecast_results,
        unit_cost=unit_cost, holding_cost=holding_cost, shortage_cost=shortage_cost,
        max_budget=10**9, max_capacity=10**9, integer=False
    )

    assert result["orders"]["Produit"] == pytest.approx(850, abs=1e-3)  # 1000 - 150


def test_order_snaps_to_upper_for_high_critical_ratio():
    """CR très élevé (rupture très coûteuse) -> mieux vaut sur-commander
    -> la commande optimale tombe sur le scénario optimiste (upper)."""

    unit_cost, holding_cost, shortage_cost = 10, 5, 300  # CR = (300-10)/(300+5) ≈ 0.951 > 0.95

    forecast_results = {"Produit": _fake_forecast_result(mean=1000, half_width=150)}

    result = _optimize(
        forecast_results,
        unit_cost=unit_cost, holding_cost=holding_cost, shortage_cost=shortage_cost,
        max_budget=10**9, max_capacity=10**9, integer=False
    )

    assert result["orders"]["Produit"] == pytest.approx(1150, abs=1e-3)  # 1000 + 150


# ==========================================================
# evaluate_strategies (WS / HN / EV / EVPI / VSS), HN via L-shaped
# ==========================================================

def test_evaluate_strategies_respects_ws_hn_ev_ordering():
    forecast_results = {
        "A": _fake_forecast_result(mean=1000, half_width=150),
        "B": _fake_forecast_result(mean=500, half_width=100),
    }

    result = evaluate_strategies(
        forecast_results,
        unit_cost=UNIT_COST, holding_cost=HOLDING_COST, shortage_cost=SHORTAGE_COST,
        max_budget=10**9, max_capacity=10**9
    )

    assert result["hn"]["status"] == "Optimal"
    assert result["hn"]["method"] == "L-shaped"
    assert result["checks_ok"]
    assert result["ws_cost"] <= result["hn"]["total_cost"] + 1e-6
    assert result["hn"]["total_cost"] <= result["ev_cost"] + 1e-6
    assert result["evpi"] >= -1e-6
    assert result["vss"] >= -1e-6


def test_evaluate_strategies_with_binding_capacity():
    """Les inégalités WS <= HN <= EV doivent tenir aussi quand la
    capacité est la contrainte active (pas seulement en marché libre)."""

    forecast_results = {
        "A": _fake_forecast_result(mean=1000, half_width=150),
        "B": _fake_forecast_result(mean=1000, half_width=100),
    }

    result = evaluate_strategies(
        forecast_results,
        unit_cost=UNIT_COST, holding_cost=HOLDING_COST, shortage_cost=SHORTAGE_COST,
        max_budget=10**9, max_capacity=800
    )

    assert result["checks_ok"]


# ==========================================================
# solve_optimal : dispatch PDE direct / L-shaped (utilitaire général,
# indépendant du choix "toujours L-shaped" fait par evaluate_strategies)
# ==========================================================

def test_solve_optimal_dispatches_to_pde_direct_below_threshold():
    forecast_results = {"A": _fake_forecast_result(mean=1000, half_width=150)}
    scenarios, probabilities = generate_demand_scenarios(forecast_results)

    result = solve_optimal(
        scenarios, probabilities,
        unit_cost=UNIT_COST, holding_cost=HOLDING_COST, shortage_cost=SHORTAGE_COST,
        max_budget=10**9, max_capacity=10**9, lshaped_threshold=100
    )

    assert result["method"] == "PDE direct"
    assert result["status"] == "Optimal"


def test_solve_optimal_dispatches_to_lshaped_above_threshold():
    """Force le seuil très bas (1) pour vérifier le branchement
    L-shaped sans avoir besoin de 100 produits réels dans le test."""

    forecast_results = {
        "A": _fake_forecast_result(mean=1000, half_width=150),
        "B": _fake_forecast_result(mean=500, half_width=100),
    }
    scenarios, probabilities = generate_demand_scenarios(forecast_results)

    direct = solve_recourse_problem(
        scenarios, probabilities,
        unit_cost=UNIT_COST, holding_cost=HOLDING_COST, shortage_cost=SHORTAGE_COST,
        max_budget=10**9, max_capacity=10**9
    )
    dispatched = solve_optimal(
        scenarios, probabilities,
        unit_cost=UNIT_COST, holding_cost=HOLDING_COST, shortage_cost=SHORTAGE_COST,
        max_budget=10**9, max_capacity=10**9, lshaped_threshold=1
    )

    assert dispatched["method"] == "L-shaped"
    assert dispatched["status"] in ("Optimal", "MaxIterations")
    # Complété par evaluate_fixed_order -- doit avoir les mêmes clés que le PDE direct
    assert set(dispatched) >= {"expected_shortage", "expected_surplus", "service_level"}
    assert dispatched["total_cost"] == pytest.approx(direct["total_cost"], rel=0.01)


# ==========================================================
# Stratégie Naïve (moyenne historique) + tableau de comparaison
# ==========================================================

def test_evaluate_strategies_with_naive_baseline():
    forecast_results = {
        "A": _fake_forecast_result(mean=1000, half_width=150),
        "B": _fake_forecast_result(mean=500, half_width=100),
    }
    # Une commande naïve délibérément mauvaise (bien en dessous de la
    # moyenne) pour vérifier qu'elle ressort bien comme la plus chère.
    historical_orders = {"A": 400, "B": 200}

    result = evaluate_strategies(
        forecast_results,
        unit_cost=UNIT_COST, holding_cost=HOLDING_COST, shortage_cost=SHORTAGE_COST,
        max_budget=10**9, max_capacity=10**9,
        historical_orders=historical_orders
    )

    assert result["naive_cost"] is not None
    assert result["naive_orders"] == historical_orders
    assert result["naive_cost"] >= result["hn"]["total_cost"]

    strategies_present = {row["strategy"] for row in result["comparison"]}
    assert strategies_present == {"WS", "HN", "EV", "Naïf"}

    # Trié par coût croissant
    costs = [row["cost"] for row in result["comparison"]]
    assert costs == sorted(costs)

    # HN est la référence (vs_hn_abs=0, vs_hn_pct=0)
    hn_row = next(row for row in result["comparison"] if row["strategy"] == "HN")
    assert hn_row["vs_hn_abs"] == pytest.approx(0.0, abs=1e-6)
    assert hn_row["vs_hn_pct"] == pytest.approx(0.0, abs=1e-6)

    naive_row = next(row for row in result["comparison"] if row["strategy"] == "Naïf")
    assert naive_row["vs_hn_abs"] == pytest.approx(
        result["naive_cost"] - result["hn"]["total_cost"]
    )


def test_naive_orders_exceeding_capacity_are_scaled_down():
    """
    Non-régression : la commande naïve (moyenne historique) ne connaît
    pas le budget/la capacité et peut les dépasser. Sans réduction, son
    coût "évalué" ignore la contrainte de capacité et ressort
    artificiellement moins cher que HN (qui la respecte) -- un plan
    physiquement impossible ne doit jamais sembler "moins cher".
    """

    forecast_results = {
        "A": _fake_forecast_result(mean=1000, half_width=100),
        "B": _fake_forecast_result(mean=1000, half_width=100),
    }
    # Somme = 2000, largement au-dessus de la capacité choisie (800)
    historical_orders = {"A": 1000, "B": 1000}

    result = evaluate_strategies(
        forecast_results,
        unit_cost=UNIT_COST, holding_cost=HOLDING_COST, shortage_cost=SHORTAGE_COST,
        max_budget=10**9, max_capacity=800,
        historical_orders=historical_orders
    )

    assert result["naive_scale_applied"] == pytest.approx(800 / 2000)
    assert sum(result["naive_orders"].values()) == pytest.approx(800)
    # La commande naïve réduite doit rester cohérente avec les
    # proportions d'origine (1:1 ici)
    assert result["naive_orders"]["A"] == pytest.approx(result["naive_orders"]["B"])


def test_evaluate_strategies_without_naive_baseline_omits_it():
    forecast_results = {"A": _fake_forecast_result(mean=1000, half_width=150)}

    result = evaluate_strategies(
        forecast_results,
        unit_cost=UNIT_COST, holding_cost=HOLDING_COST, shortage_cost=SHORTAGE_COST,
        max_budget=10**9, max_capacity=10**9
    )

    assert result["naive_cost"] is None
    assert {row["strategy"] for row in result["comparison"]} == {"WS", "HN", "EV"}


# ==========================================================
# solve_lshaped (décomposition de Benders) vs PDE direct
# ==========================================================

@pytest.mark.parametrize("max_capacity", [10**9, 1500])
def test_lshaped_matches_direct_pde(max_capacity):
    forecast_results = {
        "A": _fake_forecast_result(mean=1000, half_width=150),
        "B": _fake_forecast_result(mean=500, half_width=100),
        "C": _fake_forecast_result(mean=700, half_width=120),
    }

    scenarios, probabilities = generate_demand_scenarios(forecast_results)

    direct = solve_recourse_problem(
        scenarios, probabilities,
        unit_cost=UNIT_COST, holding_cost=HOLDING_COST, shortage_cost=SHORTAGE_COST,
        max_budget=10**9, max_capacity=max_capacity
    )

    lshaped = solve_lshaped(
        scenarios, probabilities,
        unit_cost=UNIT_COST, holding_cost=HOLDING_COST, shortage_cost=SHORTAGE_COST,
        max_budget=10**9, max_capacity=max_capacity, tol=0.001
    )

    assert direct["status"] == "Optimal"
    assert lshaped["status"] == "Optimal"
    assert sum(lshaped["orders"].values()) <= max_capacity + 1e-6
    assert lshaped["total_cost"] == pytest.approx(direct["total_cost"], rel=0.01)


# ==========================================================
# baselines.compute_unit_costs (prix réel par produit)
# ==========================================================

def _make_df_with_price():
    return pd.DataFrame({
        "product": ["A", "A", "B", "B"],
        "quantity": [10, 12, 5, 6],
        "unit_price": [9.5, 9.5, 25.0, 25.0],
    })


def test_compute_unit_costs_uses_real_price_per_product():
    costs = compute_unit_costs(_make_df_with_price(), default_unit_cost=1.0)

    assert costs == {"A": pytest.approx(9.5), "B": pytest.approx(25.0)}


def test_compute_unit_costs_falls_back_without_price_column():
    df_no_price = _make_df_with_price().drop(columns=["unit_price"])

    costs = compute_unit_costs(df_no_price, default_unit_cost=7.0)

    assert costs == {"A": 7.0, "B": 7.0}


def test_compute_unit_costs_can_be_used_directly_by_the_optimizer():
    """Vérifie que le dict retourné s'intègre tel quel dans solve_optimal
    (cᵢ par produit, cf. model.build_stochastic_model)."""

    forecast_results = {
        "A": _fake_forecast_result(mean=1000, half_width=100),
        "B": _fake_forecast_result(mean=1000, half_width=100),
    }
    unit_cost = compute_unit_costs(_make_df_with_price(), default_unit_cost=1.0)

    scenarios, probabilities = generate_demand_scenarios(forecast_results)
    result = solve_optimal(
        scenarios, probabilities,
        unit_cost=unit_cost, holding_cost=HOLDING_COST, shortage_cost=SHORTAGE_COST,
        max_budget=10**9, max_capacity=800
    )

    assert result["status"] == "Optimal"
    # Le produit le moins cher (A, 9.5) doit être favorisé par rapport
    # au plus cher (B, 25.0) sous une capacité contrainte
    assert result["orders"]["A"] > result["orders"]["B"]
