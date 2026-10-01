import numpy as np
import pulp


# Probabilités des 3 scénarios (pessimiste / médian / optimiste),
# cohérentes avec des quantiles à 5% / 50% / 95% (ex : confidence_level
# = 0.90 côté forecasting -- cf. prophet_model.forecast_prophet et
# ml_model.forecast_xgboost). Le "5%" de masse en dessous de `lower`
# est concentré sur le scénario pessimiste, le "5%" au-dessus de
# `upper` sur l'optimiste, et les 90% restants sur la prédiction
# centrale -- une discrétisation minimale à 3 points, directement
# ancrée sur ce que les modèles ont déjà calculé.
SCENARIO_PROBABILITIES = np.array([0.05, 0.90, 0.05])


def generate_demand_scenarios(forecast_results: dict) -> tuple[dict, np.ndarray]:
    """
    Construit 3 scénarios de demande DIRECTEMENT à partir des quantiles
    déjà calculés par le forecasting (Prophet ou XGBoost) -- pas de
    nouvelle hypothèse de distribution, pas de ré-échantillonnage.

    Pour chaque produit :
        scénario "pessimiste" : lower        (borne basse du modèle)
        scénario "médian"     : predictions   (prédiction ponctuelle)
        scénario "optimiste"  : upper         (borne haute du modèle)

    avec probabilités [0.05, 0.90, 0.05] (cf. SCENARIO_PROBABILITIES),
    cohérentes avec des quantiles à 5%/95% -- peu importe que
    'lower'/'upper' viennent de Prophet (interval_width) ou de XGBoost
    (régression quantile), du moment qu'ils représentent les mêmes
    niveaux de confiance.

    Les 3 scénarios sont CORRÉLÉS entre produits : l'indice 0
    représente TOUS les produits à leur borne basse simultanément
    (choc de marché commun), l'indice 2 tous à leur borne haute, etc.

    Ne fonctionne QUE pour un horizon de 1 mois : chaque résultat de
    forecast doit contenir exactement une valeur dans 'predictions'/
    'lower'/'upper' -- le pipeline (forecasting -> optimisation)
    raisonne mois par mois, pas sur un horizon agrégé.

    Parameters
    ----------
    forecast_results : dict
        {produit: résultat de forecast_prophet / forecast_xgboost /
        select_best_model}, avec 'predictions', 'lower', 'upper' -
        chacun une liste à UN seul élément (horizon=1).

    Returns
    -------
    (scenarios, probabilities)
        scenarios : dict {produit: np.ndarray de shape (3,)}.
        probabilities : np.ndarray de shape (3,) = [0.05, 0.90, 0.05].
    """

    scenarios = {}

    for product, result in forecast_results.items():

        predictions = result["predictions"]
        lower = result["lower"]
        upper = result["upper"]

        if len(predictions) != 1:
            raise ValueError(
                f"generate_demand_scenarios suppose un horizon de 1 mois, "
                f"mais '{product}' a {len(predictions)} valeur(s) prédite(s). "
                "Relance le forecasting avec horizon=1."
            )

        scenarios[product] = np.array([
            max(float(lower[0]), 0),
            max(float(predictions[0]), 0),
            max(float(upper[0]), 0)
        ])

    return scenarios, SCENARIO_PROBABILITIES.copy()


def add_recourse_constraints(
    problem: pulp.LpProblem,
    order_vars: dict,
    surplus_vars: dict,
    shortage_vars: dict,
    scenarios: dict
) -> None:
    """
    Ajoute, pour chaque produit p et chaque scénario s, la contrainte
    de recours qui relie la commande à la demande réalisée :

        order[p] - demande[p, s] = surplus[p, s] - shortage[p, s]

    surplus[p, s] et shortage[p, s] sont des variables >= 0 : si la
    demande dépasse la commande, shortage absorbe l'écart (rupture) ;
    sinon, surplus l'absorbe (stock excédentaire).
    """

    for product, demand_scenarios in scenarios.items():
        for s, demand in enumerate(demand_scenarios):
            problem += (
                order_vars[product] - demand
                == surplus_vars[(product, s)] - shortage_vars[(product, s)],
                f"recours_{product}_{s}"
            )


def add_budget_constraint(
    problem: pulp.LpProblem,
    order_vars: dict,
    unit_cost_by_product: dict,
    max_budget: float
) -> None:
    """Le coût d'achat total (Σᵢ cᵢqᵢ) ne doit pas dépasser le budget
    disponible."""

    problem += (
        pulp.lpSum(
            order_vars[product] * unit_cost_by_product[product]
            for product in order_vars
        ) <= max_budget,
        "budget_max"
    )


def add_capacity_constraint(
    problem: pulp.LpProblem,
    order_vars: dict,
    max_capacity: float
) -> None:
    """La quantité totale commandée (tous produits confondus) ne doit
    pas dépasser la capacité de stockage disponible."""

    problem += (
        pulp.lpSum(order_vars.values()) <= max_capacity,
        "capacite_max"
    )
