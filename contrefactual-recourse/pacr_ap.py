"""PACR-AP — Plausible Actionable Counterfactual Recourse with Action Paths.

Méthode hybride de recourse contrefactuel par chemins d'actions plausibles,
adaptée au credit risk scoring. Le notebook
``notebooks/hybrid-counterfactual-recourse.ipynb`` consomme ce module pour
toute l'infrastructure ; il ne contient que la définition pédagogique de la
méthode, le chargement des données et l'orchestration des audits.

Briques empruntées à la littérature
-----------------------------------
- Wachter, Mittelstadt & Russell (2017) — formulation contrefactuelle
- Ustun, Spangher & Liu (2019) — schéma d'actionnabilité (mutable/immutable)
- Karimi et al. (2020, MACE) — exploration discrète de l'espace d'actions
- Poyiadzi et al. (2020, FACE) — plausibilité de **trajectoire** (pas juste endpoint)
- Breunig, Kriegel, Ng & Sander (2000, LOF) — plausibilité **conjointe**
- Hardt, Price & Srebro (2016) — métriques de fairness inspirées de EOpp
- Karimi et al. (2021) — distinction prédictif / causal (garde-fou)

Adaptations propres à PACR-AP
-----------------------------
- Seuil cost-optimal $\\tau^* = C_{FP}/(C_{FP}+C_{FN})$ (Elkan, 2001)
- Coûts d'actions calibrés empiriquement (sigma ou information-théorique)
- Schéma d'actionnabilité externalisé en YAML, négociable, avec variants
- Mining des magnitudes depuis les transitions refusé → favorable observées
- Plausibilité hybride conjonctive (marginale NN ∧ joint LOF)
- Audit *fairness of recourse* par groupe sensible
- Triple stress-test du verdict (schéma, magnitudes, plausibilité)

Référence détaillée : ``docs/action-path-recourse.md``.

Organisation du module
----------------------
1. Helpers : prédiction, groupe sensible, signature d'état
2. DecisionRule et ses sous-classes
3. FeatureSpec et chargement YAML
4. Validateurs d'actionnabilité et coûts empiriques
5. Action et factories par défaut
6. Mining d'actions depuis les données
7. Plausibilité marginale (NN) et calibration
8. Plausibilité conjointe (LOF) — extension §17 du notebook
9. PlausibilityConfig
10. Graphe d'actions local (BFS + beam search)
11. Robustesse bootstrap
12. Path, scoring multi-objectif et sélection diverse
13. API principale
14. Audit fairness *of recourse*
15. Plotting helpers
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# sklearn / xgboost imports kept local to functions that need them, to keep the
# module loadable even when optional deps are missing.


# ============================================================================
# 1. Helpers
# ============================================================================

def predict_score(model, x_df: pd.DataFrame,
                  feature_cols: Optional[List[str]] = None) -> np.ndarray:
    """Probabilité prédite de la classe favorable Y=1.

    Wrapper léger autour de ``model.predict_proba`` qui sélectionne les colonnes
    attendues et renvoie la probabilité de la classe positive.

    Parameters
    ----------
    model : sklearn-compatible Pipeline
        Le classifieur entraîné, supportant ``predict_proba``.
    x_df : pd.DataFrame
        Un ou plusieurs états dans l'espace human-readable.
    feature_cols : list of str, optional
        Colonnes à sélectionner avant prédiction. Si ``None``, toutes les
        colonnes de ``x_df`` sont utilisées.

    Returns
    -------
    np.ndarray of shape (n,)
        Probabilités de classe favorable pour chaque ligne.
    """
    if feature_cols is None:
        feature_cols = list(x_df.columns)
    return model.predict_proba(x_df[feature_cols])[:, 1]


def get_group(x_df: pd.DataFrame, age_threshold: float = 25.0) -> np.ndarray:
    """Attribut sensible A binarisé : A=1 adultes (favorisé), A=0 jeunes (défavorisé).

    Convention équipe : seuil = 25 ans, codage standard fairness (1 = favorisé).

    Parameters
    ----------
    x_df : pd.DataFrame
        Doit contenir une colonne ``age``.
    age_threshold : float, default 25.0
        Seuil de binarisation.

    Returns
    -------
    np.ndarray of int (0 ou 1)
    """
    return (x_df["age"].astype(float).values >= age_threshold).astype(int)


def state_signature(state: pd.Series) -> str:
    """Signature canonique d'un état pour la déduplication dans le graphe."""
    return "|".join(f"{c}={state[c]}" for c in sorted(state.index))


# ============================================================================
# 2. DecisionRule — abstraction du seuillage
# ============================================================================

class DecisionRule:
    """Interface abstraite pour une règle de décision $D : (s, a) \\to \\{0, 1\\}$.

    Toute la suite du pipeline appelle ``rule.is_favorable(score, group, model_name)``
    sans hypothèse sur la nature du seuillage. Permet de basculer entre seuil
    global, seuil par groupe, seuils Hardt post-processing, ou règles custom.
    """
    name: str = "abstract"

    def is_favorable(self, score: float, group: Optional[int] = None,
                     model_name: Optional[str] = None) -> bool:
        raise NotImplementedError

    def required_threshold(self, group: Optional[int] = None,
                           model_name: Optional[str] = None) -> float:
        raise NotImplementedError


@dataclass
class GlobalThresholdRule(DecisionRule):
    """Seuil global unique $\\tau$ (avec marge optionnelle).

    Attributes
    ----------
    tau : float
        Seuil de décision. Pour le credit scoring cost-sensitive, utiliser
        ``tau = C_FP / (C_FP + C_FN)`` (Elkan, 2001).
    margin : float, default 0.0
        Marge de sécurité au-dessus du seuil nominal.
    """
    tau: float = 0.5
    margin: float = 0.0
    name: str = "global"

    def is_favorable(self, score, group=None, model_name=None):
        return score >= self.tau + self.margin

    def required_threshold(self, group=None, model_name=None):
        return self.tau + self.margin


@dataclass
class GroupThresholdRule(DecisionRule):
    """Seuils $\\tau_a$ différents par groupe sensible."""
    tau_by_group: Dict[int, float] = field(default_factory=lambda: {0: 0.5, 1: 0.5})
    margin: float = 0.0
    name: str = "group"

    def is_favorable(self, score, group=None, model_name=None):
        assert group is not None, "Group requis pour GroupThresholdRule"
        return score >= self.tau_by_group[int(group)] + self.margin

    def required_threshold(self, group=None, model_name=None):
        return self.tau_by_group[int(group)] + self.margin


@dataclass
class HardtStrictRule(DecisionRule):
    """Seuils stricts par groupe issus du post-processing de Hardt et al. (2016).

    Notes
    -----
    Le seuil strict est la borne haute de la bande de randomisation Hardt :
    au-dessus du strict on accepte systématiquement.
    """
    tau_strict_by_group: Dict[int, float] = field(default_factory=dict)
    margin: float = 0.0
    name: str = "hardt_strict"

    def is_favorable(self, score, group=None, model_name=None):
        return score >= self.tau_strict_by_group[int(group)] + self.margin

    def required_threshold(self, group=None, model_name=None):
        return self.tau_strict_by_group[int(group)] + self.margin


@dataclass
class HardtLooseRule(DecisionRule):
    """Seuils bas Hardt — zone de randomisation uniquement.

    Notes
    -----
    Cible faible : un chemin qui n'atteint que le seuil loose entre seulement
    dans la bande randomisée. À éviter comme objectif de recourse.
    """
    tau_loose_by_group: Dict[int, float] = field(default_factory=dict)
    margin: float = 0.0
    name: str = "hardt_loose"

    def is_favorable(self, score, group=None, model_name=None):
        return score >= self.tau_loose_by_group[int(group)] + self.margin

    def required_threshold(self, group=None, model_name=None):
        return self.tau_loose_by_group[int(group)] + self.margin


@dataclass
class CustomDecisionRule(DecisionRule):
    """Règle de décision arbitraire fournie sous forme de fonctions."""
    is_favorable_fn: Callable = None
    required_threshold_fn: Callable = None
    name: str = "custom"

    def is_favorable(self, score, group=None, model_name=None):
        return bool(self.is_favorable_fn(score, group, model_name))

    def required_threshold(self, group=None, model_name=None):
        return float(self.required_threshold_fn(group, model_name))


def cost_optimal_tau(cost_fp: float = 5.0, cost_fn: float = 1.0) -> float:
    """Seuil cost-optimal de Bayes : $\\tau^* = C_{FP}/(C_{FP}+C_{FN})$.

    Parameters
    ----------
    cost_fp : float, default 5.0
        Coût d'un faux positif (approuver un mauvais). German Credit = 5.
    cost_fn : float, default 1.0
        Coût d'un faux négatif (refuser un bon). German Credit = 1.

    Returns
    -------
    float
        Le seuil qui minimise le coût attendu (Elkan, 2001).
    """
    return cost_fp / (cost_fp + cost_fn)


# ============================================================================
# 3. FeatureSpec — Schéma d'actionnabilité
# ============================================================================

@dataclass
class FeatureSpec:
    """Annotation d'une feature pour le recourse.

    Encodage de la politique d'actionnabilité par feature. Le schéma complet
    (un FeatureSpec par feature) est externalisé en YAML pour rester
    négociable par des stakeholders non-développeurs.

    Attributes
    ----------
    name : str
        Nom de la feature (doit matcher une colonne du dataset).
    mutable : bool
        L'individu peut-il agir sur cette feature dans l'horizon de recourse ?
    direction : {"any", "increase_only", "decrease_only", "fixed"}
        Sens de modification autorisé. ``"fixed"`` = immutable strict.
    type : {"continuous", "ordinal", "categorical", "binary"}
    allowed_values : list, optional
        Pour les ordinales/catégorielles, liste exhaustive des valeurs licites.
    rationale : str, default ""
        Justification textuelle du choix (apparaît dans l'audit).

    Notes
    -----
    Les coûts ne sont **pas** stockés dans FeatureSpec : ils sont dérivés
    empiriquement par les fonctions ``empirical_*_cost`` (en multiples de σ
    pour les continues, information-théoriques pour les ordinales/catégorielles).
    """
    name: str
    mutable: bool
    direction: str = "any"
    type: str = "continuous"
    allowed_values: Optional[List] = None
    rationale: str = ""


def load_schema_from_yaml(yaml_path: Union[str, Path],
                           variant_name: str = "conservateur"
                           ) -> Tuple[Dict[str, FeatureSpec], Dict[str, List], Dict[str, Dict]]:
    """Charge le schéma d'actionnabilité depuis un fichier YAML.

    Le YAML déclare une ``base_features`` (politique par défaut) et des
    ``variants`` (overrides ciblés pour analyse de sensibilité).

    Parameters
    ----------
    yaml_path : str or Path
        Chemin vers le fichier YAML (typiquement ``docs/schema-actions.yaml``).
    variant_name : str, default "conservateur"
        Variant à charger. Doit exister dans la section ``variants`` du YAML.

    Returns
    -------
    schema : dict {feature_name → FeatureSpec}
    ordinal_orders : dict {feature_name → list of allowed values in order}
        Pour les features ordinales seulement.
    empirical_bounds : dict
        Configuration des bornes empiriques pour les préconditions.
    """
    import yaml
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    if variant_name not in data["variants"]:
        raise ValueError(f"Variant inconnu : {variant_name!r}. "
                          f"Disponibles : {list(data['variants'].keys())}")
    base = data["base_features"]
    overrides = data["variants"][variant_name].get("overrides") or {}

    schema: Dict[str, FeatureSpec] = {}
    for fname, base_spec in base.items():
        merged = {**base_spec}
        if fname in overrides:
            merged.update(overrides[fname])
        schema[fname] = FeatureSpec(
            name=fname,
            mutable=bool(merged.get("mutable", False)),
            direction=merged.get("direction", "fixed"),
            type=merged.get("type", "continuous"),
            allowed_values=merged.get("allowed_values"),
            rationale=merged.get("rationale", ""),
        )
    ordinal_orders = {
        f: list(spec.allowed_values)
        for f, spec in schema.items()
        if spec.type == "ordinal" and spec.allowed_values is not None
    }
    empirical_bounds = data.get("empirical_bounds", {})
    return schema, ordinal_orders, empirical_bounds


# ============================================================================
# 4. Validateurs d'actionnabilité + coûts empiriques
# ============================================================================

def is_actionable_change(x_orig: pd.Series, x_new: pd.Series,
                          schema: Dict[str, FeatureSpec]) -> bool:
    """Teste qu'aucune feature immuable n'a changé et que toutes les directions
    sont respectées.

    Parameters
    ----------
    x_orig, x_new : pd.Series
        États avant et après transition.
    schema : dict {feature_name → FeatureSpec}

    Returns
    -------
    bool
        ``True`` ssi tous les changements respectent le schéma (mutabilité,
        direction, valeurs autorisées).
    """
    for fname, spec in schema.items():
        if fname not in x_orig.index:
            continue
        v_o, v_n = x_orig[fname], x_new[fname]
        if v_o == v_n:
            continue
        if not spec.mutable or spec.direction == "fixed":
            return False
        if spec.allowed_values is not None and v_n not in spec.allowed_values:
            return False
        if spec.type == "continuous":
            if spec.direction == "increase_only" and float(v_n) < float(v_o):
                return False
            if spec.direction == "decrease_only" and float(v_n) > float(v_o):
                return False
        if spec.type == "ordinal" and spec.allowed_values is not None:
            i_o = spec.allowed_values.index(v_o)
            i_n = spec.allowed_values.index(v_n)
            if spec.direction == "increase_only" and i_n < i_o:
                return False
            if spec.direction == "decrease_only" and i_n > i_o:
                return False
    return True


def empirical_continuous_cost(feature: str, delta: float,
                               X_train: pd.DataFrame) -> float:
    """Coût d'un changement Δ pour une feature continue, en multiples de σ.

    $$c = |\\Delta| / \\sigma_{train}$$

    Interprétation : « cette action vaut k écarts-types de la population
    observée ». Indépendant de l'unité de la feature.
    """
    sigma = float(X_train[feature].astype(float).std())
    return abs(float(delta)) / max(sigma, 1.0)


def empirical_ordinal_cost(feature: str, target_value: Any,
                            X_train: pd.DataFrame,
                            orders: Dict[str, List]) -> float:
    """Coût information-théorique pour atteindre ``target_value`` ordinal.

    $$c = -\\log P(\\text{level} \\geq \\text{target\\_value})$$

    Interprétation : un niveau cible rare coûte plus cher qu'un niveau cible
    fréquent. En nats (logarithme népérien).
    """
    order = orders[feature]
    target_idx = order.index(target_value)
    p = X_train[feature].apply(
        lambda v: order.index(v) >= target_idx if v in order else False
    ).mean()
    return float(-np.log(max(p, 1e-3)))


def empirical_categorical_cost(feature: str, target_value: Any,
                                X_train: pd.DataFrame) -> float:
    """Coût information-théorique pour atteindre la modalité catégorielle cible.

    $$c = -\\log P(\\text{feature} = \\text{target\\_value})$$
    """
    p = float((X_train[feature] == target_value).mean())
    return float(-np.log(max(p, 1e-3)))


def action_cost(x_orig: pd.Series, x_new: pd.Series,
                 schema: Dict[str, FeatureSpec], X_train: pd.DataFrame,
                 orders: Optional[Dict[str, List]] = None) -> float:
    """Coût total empirique de la transition cumulée $x_{orig} \\to x_{new}$.

    Somme par feature des coûts élémentaires (continus, ordinaux ou
    catégoriels). Indépendant du chemin emprunté.

    Notes
    -----
    Cette fonction n'est pas utilisée dans le pipeline path-based principal
    (qui somme les coûts pas-à-pas via ``Action.cost_fn``). Utile pour
    comparer des contrefactuels ponctuels.
    """
    if orders is None:
        orders = {f: s.allowed_values for f, s in schema.items()
                  if s.type == "ordinal" and s.allowed_values}
    total = 0.0
    for fname, spec in schema.items():
        if fname not in x_orig.index:
            continue
        v_o, v_n = x_orig[fname], x_new[fname]
        if v_o == v_n:
            continue
        if spec.type == "continuous":
            total += empirical_continuous_cost(fname, float(v_n) - float(v_o), X_train)
        elif spec.type == "ordinal" and spec.allowed_values is not None:
            order = spec.allowed_values
            higher = v_n if order.index(v_n) > order.index(v_o) else v_o
            total += empirical_ordinal_cost(fname, higher, X_train, {fname: order})
        else:
            total += empirical_categorical_cost(fname, v_n, X_train)
    return total


def changed_features(x_orig: pd.Series, x_new: pd.Series,
                      schema: Dict[str, FeatureSpec]) -> Dict[str, Tuple]:
    """Dictionnaire {feature : (ancienne valeur, nouvelle valeur)} des changements."""
    out = {}
    for fname in schema:
        if fname in x_orig.index and x_orig[fname] != x_new[fname]:
            out[fname] = (x_orig[fname], x_new[fname])
    return out


# ============================================================================
# 5. Action et factories par défaut
# ============================================================================

@dataclass
class Action:
    """Action métier élémentaire transformant un état en un autre.

    Chaque action encapsule trois fonctions pures qui garantissent par
    construction le respect du schéma d'actionnabilité (aucune action ne peut
    modifier une feature immuable).

    Attributes
    ----------
    name : str
        Identifiant unique (e.g. ``"reduce_amount_10pct"``).
    feature : str
        Feature principalement modifiée par l'action.
    description : str
        Texte humain pour l'explication finale.
    apply_fn : callable (pd.Series → pd.Series or None)
        Applique l'action. Retourne ``None`` si non-applicable.
    cost_fn : callable (pd.Series → float)
        Coût empirique de l'action depuis cet état.
    precondition_fn : callable (pd.Series → bool)
        Préfiltre rapide : l'action est-elle applicable depuis cet état ?
    action_type : str
        Type sémantique (``"decrease_amount"``, ``"improve_ordinal"``, ...).
    """
    name: str
    feature: str
    description: str
    apply_fn: Callable[[pd.Series], Optional[pd.Series]]
    cost_fn: Callable[[pd.Series], float]
    precondition_fn: Callable[[pd.Series], bool]
    action_type: str = "generic"


def make_decrease_amount_action(pct: float, schema, X_train, bounds) -> Action:
    """Factory : réduire ``credit_amount`` de ``pct * 100`` %.

    Bornes : la valeur résultante doit rester ≥ percentile 1 du training set.
    """
    spec = schema["credit_amount"]
    min_amt = float(X_train["credit_amount"].quantile(
        bounds["credit_amount"]["min_quantile"]))

    def apply_fn(x):
        new_amt = float(x["credit_amount"]) * (1 - pct)
        if new_amt < min_amt:
            return None
        out = x.copy(); out["credit_amount"] = new_amt
        return out

    def cost_fn(x):
        return empirical_continuous_cost("credit_amount",
                                          float(x["credit_amount"]) * pct, X_train)

    def precondition_fn(x):
        return (spec.mutable and "credit_amount" in x.index
                and float(x["credit_amount"]) * (1 - pct) >= min_amt)

    return Action(
        name=f"reduce_amount_{int(pct * 100)}pct",
        feature="credit_amount",
        description=f"réduire le montant du crédit de {int(pct * 100)} %",
        apply_fn=apply_fn, cost_fn=cost_fn, precondition_fn=precondition_fn,
        action_type="decrease_amount",
    )


def make_decrease_duration_action(delta_months: int, schema, X_train, bounds) -> Action:
    """Factory : raccourcir ``duration`` de ``delta_months``."""
    spec = schema["duration"]
    min_dur = max(1, float(X_train["duration"].quantile(
        bounds["duration"]["min_quantile"])))

    def apply_fn(x):
        new_d = int(float(x["duration"])) - delta_months
        if new_d < min_dur:
            return None
        out = x.copy(); out["duration"] = new_d
        return out

    def cost_fn(x):
        return empirical_continuous_cost("duration", float(delta_months), X_train)

    def precondition_fn(x):
        return (spec.mutable and "duration" in x.index
                and int(float(x["duration"])) - delta_months >= min_dur)

    return Action(
        name=f"reduce_duration_{delta_months}m",
        feature="duration",
        description=f"raccourcir la durée du prêt de {delta_months} mois",
        apply_fn=apply_fn, cost_fn=cost_fn, precondition_fn=precondition_fn,
        action_type="decrease_duration",
    )


def make_improve_ordinal_action(feature: str, schema, X_train,
                                  orders: Dict[str, List]) -> Action:
    """Factory : améliorer une feature ordinale d'un cran (+1 dans l'ordre)."""
    spec = schema[feature]
    order = orders[feature]

    def apply_fn(x):
        cur = x[feature]
        if cur not in order:
            return None
        i = order.index(cur)
        if i + 1 >= len(order):
            return None
        out = x.copy(); out[feature] = order[i + 1]
        return out

    def cost_fn(x):
        cur = x[feature]
        if cur not in order:
            return 0.0
        i = order.index(cur)
        if i + 1 >= len(order):
            return 0.0
        return empirical_ordinal_cost(feature, order[i + 1], X_train, {feature: order})

    def precondition_fn(x):
        if not spec.mutable or feature not in x.index:
            return False
        cur = x[feature]
        return cur in order and order.index(cur) + 1 < len(order)

    return Action(
        name=f"improve_{feature}",
        feature=feature,
        description=f"améliorer {feature.replace('_', ' ')} d'un cran",
        apply_fn=apply_fn, cost_fn=cost_fn, precondition_fn=precondition_fn,
        action_type="improve_ordinal",
    )


def make_add_party_action(target_val: str, schema, X_train) -> Action:
    """Factory : ajouter un garant ou co-emprunteur (modifie ``other_parties``)."""
    spec = schema["other_parties"]

    def apply_fn(x):
        if x["other_parties"] == target_val:
            return None
        out = x.copy(); out["other_parties"] = target_val
        return out

    def cost_fn(x):
        return empirical_categorical_cost("other_parties", target_val, X_train)

    def precondition_fn(x):
        return (spec.mutable and "other_parties" in x.index
                and x["other_parties"] != target_val)

    return Action(
        name=f"add_{target_val.replace(' ', '_')}",
        feature="other_parties",
        description=f"ajouter un {target_val} au dossier",
        apply_fn=apply_fn, cost_fn=cost_fn, precondition_fn=precondition_fn,
        action_type="add_party",
    )


def build_default_actions(schema: Dict[str, FeatureSpec],
                            X_train: pd.DataFrame,
                            orders: Dict[str, List],
                            bounds: Dict[str, Dict]) -> List[Action]:
    """Construit la liste d'actions par défaut depuis le schéma + bornes empiriques.

    Génère, pour chaque feature mutable du schéma, les actions canoniques
    (paliers 5/10/15/20 % pour ``credit_amount`` ; 6/12/18 mois pour ``duration`` ;
    +1 cran pour les ordinales mutables ; ajout de garant pour ``other_parties``).

    Returns
    -------
    list of Action
        Typiquement 9 actions sous le variant conservateur, 11 sous modéré,
        12 sous permissif.
    """
    actions: List[Action] = []
    if "credit_amount" in schema and schema["credit_amount"].mutable:
        for pct in [0.05, 0.10, 0.15, 0.20]:
            actions.append(make_decrease_amount_action(pct, schema, X_train, bounds))
    if "duration" in schema and schema["duration"].mutable:
        for d in [6, 12, 18]:
            actions.append(make_decrease_duration_action(d, schema, X_train, bounds))
    for f in ["checking_status", "savings_status", "employment"]:
        if f in schema and schema[f].mutable and f in orders:
            actions.append(make_improve_ordinal_action(f, schema, X_train, orders))
    if "other_parties" in schema and schema["other_parties"].mutable:
        for target in ["co applicant", "guarantor"]:
            actions.append(make_add_party_action(target, schema, X_train))
    return actions


# ============================================================================
# 6. Action mining — découvre les magnitudes depuis les transitions observées
# ============================================================================

def _pairwise_mixed_distance(X1: pd.DataFrame, X2: pd.DataFrame,
                              schema: Dict[str, FeatureSpec],
                              scales: Dict[str, float]) -> np.ndarray:
    """Distance mixte L1 pairwise entre deux DataFrames (vectorisé)."""
    n1, n2 = len(X1), len(X2)
    D = np.zeros((n1, n2))
    for col in X1.columns:
        s = scales.get(col, 1.0)
        if col in schema and schema[col].type == "continuous":
            v1 = X1[col].astype(float).values[:, None]
            v2 = X2[col].astype(float).values[None, :]
            D += np.abs(v1 - v2) / s
        else:
            v1 = np.asarray(X1[col].astype(object).values)[:, None]
            v2 = np.asarray(X2[col].astype(object).values)[None, :]
            D += (v1 != v2).astype(float) / s
    return D


def make_mined_pct_continuous_action(feature: str, pct: float, schema, X_train,
                                       bounds, direction: str = "decrease") -> Action:
    """Action continue minée paramétrée par un pourcentage de changement."""
    spec = schema[feature]
    min_q = bounds.get(feature, {}).get("min_quantile", 0.01)
    min_val = float(X_train[feature].quantile(min_q))
    sign = -1 if direction == "decrease" else +1

    def apply_fn(x):
        new_val = float(x[feature]) * (1 + sign * pct)
        if direction == "decrease" and new_val < min_val:
            return None
        out = x.copy(); out[feature] = new_val
        return out

    def cost_fn(x):
        return empirical_continuous_cost(feature, float(x[feature]) * pct, X_train)

    def precondition_fn(x):
        if not spec.mutable or feature not in x.index:
            return False
        new_val = float(x[feature]) * (1 + sign * pct)
        return (direction != "decrease") or new_val >= min_val

    verb = "réduire" if direction == "decrease" else "augmenter"
    return Action(
        name=f"mined_{direction}_{feature}_{int(round(pct * 100))}pct",
        feature=feature,
        description=f"{verb} {feature.replace('_', ' ')} de {int(round(pct * 100))} %",
        apply_fn=apply_fn, cost_fn=cost_fn, precondition_fn=precondition_fn,
        action_type="mined_pct",
    )


def make_mined_ordinal_step_action(feature: str, step: int, schema, X_train,
                                     orders: Dict[str, List]) -> Optional[Action]:
    """Action ordinale minée paramétrée par un pas de transition entier."""
    spec = schema[feature]
    order = orders.get(feature)
    if order is None:
        return None

    def apply_fn(x):
        cur = x[feature]
        if cur not in order:
            return None
        target_idx = order.index(cur) + step
        if not (0 <= target_idx < len(order)):
            return None
        out = x.copy(); out[feature] = order[target_idx]
        return out

    def cost_fn(x):
        cur = x[feature]
        if cur not in order:
            return 0.0
        target_idx = order.index(cur) + step
        if not (0 <= target_idx < len(order)):
            return 0.0
        return empirical_ordinal_cost(feature, order[target_idx], X_train, {feature: order})

    def precondition_fn(x):
        if not spec.mutable or feature not in x.index:
            return False
        cur = x[feature]
        return cur in order and 0 <= order.index(cur) + step < len(order)

    verb = "améliorer" if step > 0 else "régresser"
    safe_verb = "improve" if step > 0 else "regress"
    return Action(
        name=f"mined_{safe_verb}_{feature}_{abs(step)}step",
        feature=feature,
        description=f"{verb} {feature.replace('_', ' ')} de {abs(step)} cran(s)",
        apply_fn=apply_fn, cost_fn=cost_fn, precondition_fn=precondition_fn,
        action_type="mined_ordinal",
    )


def make_mined_categorical_action(feature: str, target_value: Any,
                                    schema, X_train) -> Action:
    """Action catégorielle minée vers une valeur cible spécifique."""
    spec = schema[feature]

    def apply_fn(x):
        if x[feature] == target_value:
            return None
        out = x.copy(); out[feature] = target_value
        return out

    def cost_fn(x):
        return empirical_categorical_cost(feature, target_value, X_train)

    def precondition_fn(x):
        return spec.mutable and feature in x.index and x[feature] != target_value

    safe = (str(target_value).replace(" ", "_").replace("<", "lt")
            .replace(">", "gt").replace("=", "eq")[:20])
    return Action(
        name=f"mined_set_{feature}_{safe}",
        feature=feature,
        description=f"changer {feature.replace('_', ' ')} → '{target_value}' (cible minée)",
        apply_fn=apply_fn, cost_fn=cost_fn, precondition_fn=precondition_fn,
        action_type="mined_categorical",
    )


def mine_actions_from_data(
    X_train: pd.DataFrame, y_train: np.ndarray,
    schema: Dict[str, FeatureSpec],
    orders: Dict[str, List],
    bounds: Dict[str, Dict],
    scales: Dict[str, float],
    k_neighbors: int = 10,
    min_support: int = 30,
    continuous_percentiles: Tuple[float, ...] = (0.25, 0.50, 0.75),
    ordinal_top_k: int = 2,
    categorical_top_k: int = 3,
) -> Tuple[List[Action], Dict[str, Any]]:
    """Mining d'actions depuis les transitions refusé → favorable observées.

    Pour chaque refusé ($y = 0$), on trouve ses ``k_neighbors`` voisins
    favorables ($y = 1$) les plus proches sous la distance mixte L1
    standardisée. On collecte les écarts feature-par-feature, on filtre selon
    le schéma, puis on discrétise en actions canoniques.

    Parameters
    ----------
    X_train, y_train : training set
    schema : politique d'actionnabilité (respectée comme filtre)
    orders, bounds, scales : helpers calibrés sur le training
    k_neighbors : int, default 10
    min_support : int, default 30
        Une action n'est minée que si elle est observée sur ≥ min_support paires.
    continuous_percentiles : tuple of floats
        Percentiles utilisés pour discrétiser les écarts continus.
    ordinal_top_k, categorical_top_k : int
        Nombre max d'actions ordinales/catégorielles à retenir par feature.

    Returns
    -------
    actions : list of Action
    report : dict
        Diagnostics par feature : nombre d'observations, percentiles observés,
        actions extraites. Utile pour vérifier la défendabilité statistique.

    Notes
    -----
    Les transitions observées sont des **corrélations**, pas des
    **interventions**. Le mining grounds les magnitudes dans le réel observé
    mais ne corrige pas la non-causalité fondamentale du recourse prédictif.
    """
    X_R = X_train[y_train == 0].reset_index(drop=True)
    X_F = X_train[y_train == 1].reset_index(drop=True)
    D = _pairwise_mixed_distance(X_R, X_F, schema, scales)
    nn_idx = np.argsort(D, axis=1)[:, :k_neighbors]

    mutable_features = [f for f, s in schema.items() if s.mutable]
    deltas = {f: [] for f in mutable_features}

    for i in range(len(X_R)):
        x_r = X_R.iloc[i]
        for j in nn_idx[i]:
            x_f = X_F.iloc[int(j)]
            for feat in mutable_features:
                spec = schema[feat]
                v_r, v_f = x_r[feat], x_f[feat]
                if v_r == v_f:
                    continue
                if spec.type == "continuous":
                    v_r_f, v_f_f = float(v_r), float(v_f)
                    if spec.direction == "decrease_only" and v_f_f >= v_r_f:
                        continue
                    if spec.direction == "increase_only" and v_f_f <= v_r_f:
                        continue
                    pct = abs(v_f_f - v_r_f) / max(abs(v_r_f), 1e-3)
                    deltas[feat].append({"type": "continuous", "pct": pct,
                                          "abs": abs(v_f_f - v_r_f)})
                elif spec.type == "ordinal" and feat in orders:
                    order = orders[feat]
                    if v_r not in order or v_f not in order:
                        continue
                    step = order.index(v_f) - order.index(v_r)
                    if spec.direction == "increase_only" and step <= 0:
                        continue
                    if spec.direction == "decrease_only" and step >= 0:
                        continue
                    deltas[feat].append({"type": "ordinal", "step": step, "to": v_f})
                else:
                    deltas[feat].append({"type": "categorical", "to": v_f})

    actions: List[Action] = []
    report = {
        "k_neighbors": k_neighbors, "min_support": min_support,
        "n_refused": len(X_R), "n_favorable": len(X_F),
        "n_pairs_total": len(X_R) * k_neighbors,
        "per_feature": {},
    }

    for feat, obs in deltas.items():
        spec = schema[feat]
        feat_rep = {"type": spec.type, "direction": spec.direction,
                    "n_observations": len(obs), "actions_mined": []}
        if len(obs) < min_support:
            feat_rep["skip"] = f"insufficient support ({len(obs)} < {min_support})"
            report["per_feature"][feat] = feat_rep
            continue
        if spec.type == "continuous":
            pcts_obs = [o["pct"] for o in obs]
            feat_rep["observed_pct_summary"] = {
                "min": float(np.min(pcts_obs)),
                "p25": float(np.percentile(pcts_obs, 25)),
                "p50": float(np.percentile(pcts_obs, 50)),
                "p75": float(np.percentile(pcts_obs, 75)),
                "max": float(np.max(pcts_obs)),
            }
            canon = sorted({round(float(np.percentile(pcts_obs, 100 * p)), 2)
                            for p in continuous_percentiles})
            canon = [p for p in canon if 0.02 <= p <= 0.50]
            for pct in canon:
                direction = "decrease" if spec.direction == "decrease_only" else "increase"
                act = make_mined_pct_continuous_action(feat, pct, schema, X_train,
                                                        bounds, direction)
                actions.append(act)
                feat_rep["actions_mined"].append({"type": "pct", "value": pct,
                                                    "direction": direction})
        elif spec.type == "ordinal":
            step_counts = Counter(o["step"] for o in obs)
            feat_rep["observed_steps"] = dict(step_counts.most_common())
            for step, count in step_counts.most_common(ordinal_top_k):
                if count < min_support / 5 or abs(step) > 2:
                    continue
                act = make_mined_ordinal_step_action(feat, step, schema, X_train, orders)
                if act is not None:
                    actions.append(act)
                    feat_rep["actions_mined"].append({"type": "step", "value": step,
                                                       "count": count})
        else:
            target_counts = Counter(o["to"] for o in obs)
            feat_rep["observed_targets"] = dict(target_counts.most_common())
            for target, count in target_counts.most_common(categorical_top_k):
                if count < min_support / 5:
                    continue
                act = make_mined_categorical_action(feat, target, schema, X_train)
                actions.append(act)
                feat_rep["actions_mined"].append({"type": "target", "value": target,
                                                   "count": count})
        report["per_feature"][feat] = feat_rep

    return actions, report


# ============================================================================
# 7. Plausibilité marginale (NN distance) + calibration de ε
# ============================================================================

def compute_feature_scales(X_train: pd.DataFrame,
                            schema: Dict[str, FeatureSpec]) -> Dict[str, float]:
    """Échelles pour la distance mixte : σ pour continues, 1 pour autres."""
    scales = {}
    for col in X_train.columns:
        if col not in schema:
            scales[col] = 1.0
            continue
        spec = schema[col]
        if spec.type == "continuous":
            s = float(X_train[col].astype(float).std())
            scales[col] = s if s > 0 else 1.0
        else:
            scales[col] = 1.0
    return scales


def mixed_distance(x: pd.Series, X: pd.DataFrame,
                    schema: Dict[str, FeatureSpec],
                    scales: Dict[str, float]) -> np.ndarray:
    """Distance L1 mixte normalisée entre ``x`` et chaque ligne de ``X``.

    Continues : $|\\Delta| / \\sigma$. Catégorielles/ordinales : Hamming
    pondéré par ``1/scale``.
    """
    d = np.zeros(len(X))
    for col in X.columns:
        s = scales.get(col, 1.0)
        if col in schema and schema[col].type == "continuous":
            d += np.abs(X[col].astype(float).values - float(x[col])) / s
        else:
            d += (X[col].values != x[col]).astype(float) / s
    return d


def nearest_neighbor_distance(x: pd.Series, X_train: pd.DataFrame,
                                schema: Dict[str, FeatureSpec],
                                scales: Dict[str, float]) -> float:
    """Distance au plus proche voisin dans ``X_train``."""
    return float(mixed_distance(x, X_train, schema, scales).min())


def local_density(x: pd.Series, X_train: pd.DataFrame, epsilon: float,
                   schema: Dict[str, FeatureSpec],
                   scales: Dict[str, float]) -> int:
    """Nombre de voisins du training à distance ≤ ``epsilon``."""
    d = mixed_distance(x, X_train, schema, scales)
    return int((d <= epsilon).sum())


def calibrate_epsilon(X_train: pd.DataFrame, schema: Dict[str, FeatureSpec],
                       scales: Dict[str, float], percentile: float = 95.0,
                       n_sample: int = 200, seed: int = 42) -> float:
    """Calibre ε comme le percentile p% des distances NN intra-train.

    On échantillonne ``n_sample`` points du training, on calcule pour chacun la
    2e plus petite distance (pour éviter d(x, x) = 0), puis on prend le
    percentile demandé.

    Parameters
    ----------
    percentile : float, default 95.0
        Plus haut = plus permissif. p95 = un état est plausible ssi sa
        distance NN est inférieure à 95 % des distances intra-train.

    Returns
    -------
    float
        Seuil ε à appliquer dans le filtre marginal.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_train), size=min(n_sample, len(X_train)), replace=False)
    nn_dists = []
    for i in idx:
        d = mixed_distance(X_train.iloc[int(i)], X_train, schema, scales)
        d_sorted = np.sort(d)
        nn_dists.append(d_sorted[1] if len(d_sorted) > 1 else d_sorted[0])
    return float(np.percentile(nn_dists, percentile))


# ============================================================================
# 8. Plausibilité conjointe (LOF) — extension §20
# ============================================================================

def build_joint_plausibility_check(
    X_train: pd.DataFrame, preproc, feature_cols: List[str],
    n_neighbors: int = 20, percentile: float = 5.0,
) -> Tuple[Callable[[pd.Series], bool], Callable[[pd.Series], float], float]:
    """Construit le filtre joint LOF + retourne le seuil calibré.

    Fit deux LOF :
    - ``novelty=False`` pour calibrer ρ_LOF (percentile des scores intra-train)
    - ``novelty=True`` pour scorer les états candidats du graphe

    Parameters
    ----------
    X_train : pd.DataFrame human-readable
    preproc : ColumnTransformer entraîné (transforme human → modèle)
    feature_cols : colonnes à sélectionner avant preprocessing
    n_neighbors : int, default 20
    percentile : float, default 5.0
        Percentile bas des scores intra-train définissant ρ. p5 = on rejette
        les 5 % les plus jointement-anormaux du training.

    Returns
    -------
    is_jointly_plausible : callable (pd.Series → bool)
        ``True`` ssi le score LOF de l'état est ≥ ρ_LOF.
    joint_score_fn : callable (pd.Series → float)
        Score LOF brut (plus élevé = plus inlier).
    rho_lof : float
        Seuil calibré.

    Notes
    -----
    Capture les dépendances **conjointes** entre features que la distance NN
    marginale ne voit pas. Référence : Breunig, Kriegel, Ng & Sander (2000),
    *LOF: Identifying Density-Based Local Outliers*.
    """
    from sklearn.neighbors import LocalOutlierFactor

    X_pre = preproc.transform(X_train[feature_cols])

    lof_calib = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=False)
    lof_calib.fit_predict(X_pre)
    rho_lof = float(np.percentile(lof_calib.negative_outlier_factor_, percentile))

    lof_novelty = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)
    lof_novelty.fit(X_pre)

    def joint_score_fn(x: pd.Series) -> float:
        x_pre = preproc.transform(pd.DataFrame([x])[feature_cols])
        return float(lof_novelty.score_samples(x_pre)[0])

    def is_jointly_plausible(x: pd.Series) -> bool:
        return joint_score_fn(x) >= rho_lof

    return is_jointly_plausible, joint_score_fn, rho_lof


@dataclass
class PlausibilityConfig:
    """Configuration des filtres de plausibilité.

    Attributes
    ----------
    epsilon_nn : float
        Seuil de plausibilité marginale (distance NN).
    min_density : int, default 0
        Densité locale minimale (0 = pas de contrainte).
    joint_check_fn : callable, optional
        Si fourni, ajoute un filtre joint (e.g. LOF). Signature : x → bool.
    rho_lof : float, optional
        Seuil de plausibilité jointe (informatif, déjà encodé dans le check_fn).
    """
    epsilon_nn: float = None
    min_density: int = 0
    joint_check_fn: Optional[Callable[[pd.Series], bool]] = None
    rho_lof: Optional[float] = None


# ============================================================================
# 9. Graphe d'actions local (BFS + beam search)
# ============================================================================

@dataclass
class GraphConfig:
    """Configuration de la construction du graphe d'actions local.

    Attributes
    ----------
    max_depth : int, default 3
        Profondeur maximale de la BFS.
    beam_width : int, default 50
        Largeur du faisceau par niveau (élagage par marge au seuil).
    max_nodes : int, default 500
        Borne dure sur la taille du graphe.
    max_cumulative_cost : float, optional
        Coût cumulé maximal autorisé.
    prune_unplausible_states : bool, default True
        Active les filtres de plausibilité (marginal + joint si fourni).
    prune_duplicate_states : bool, default True
        Élimine les nœuds dont la signature canonique est déjà visitée.
    random_state : int, default 42
    """
    max_depth: int = 3
    beam_width: int = 50
    max_nodes: int = 500
    max_cumulative_cost: Optional[float] = None
    prune_unplausible_states: bool = True
    prune_duplicate_states: bool = True
    random_state: int = 42


@dataclass
class GraphNode:
    """Nœud du graphe d'actions local."""
    idx: int
    state: pd.Series
    depth: int
    parent_idx: Optional[int]
    action_name: Optional[str]
    action_obj: Optional[Action]
    cumulative_cost: float
    score: float
    is_favorable: bool
    nn_distance: float
    density: int


def build_action_graph(
    x0: pd.Series, model, decision_rule: DecisionRule,
    schema: Dict[str, FeatureSpec], actions: List[Action],
    X_train_raw: pd.DataFrame, plaus_cfg: PlausibilityConfig,
    scales: Dict[str, float],
    graph_cfg: Optional[GraphConfig] = None,
    feature_cols: Optional[List[str]] = None,
    age_threshold: float = 25.0,
    model_name: str = "model",
) -> Dict[str, Any]:
    """Construit un graphe d'actions local par BFS + beam search.

    À chaque profondeur, on conserve les ``beam_width`` nœuds les plus
    prometteurs par marge $s - \\tau$. Quand un nœud devient favorable, le
    chemin racine → nœud est extrait et sauvegardé, et le nœud n'est pas
    re-exploré.

    Filtres appliqués à chaque transition candidate :
    1. Précondition de l'action
    2. ``is_actionable_change`` (sécurité contre violations du schéma)
    3. Déduplication par ``state_signature``
    4. Plausibilité marginale : $d_{NN}(\\mathbf{x}_t) \\leq \\varepsilon$
    5. Plausibilité jointe : ``plaus_cfg.joint_check_fn(x_t)`` si fournie
    6. Budget cumulé : ``cumulative_cost ≤ max_cumulative_cost``

    Returns
    -------
    dict with keys :
        - ``nodes`` : liste de tous les ``GraphNode`` explorés
        - ``favorable_paths`` : liste de listes d'indices (racine → favorable)
        - ``rejection_counts`` : compteurs par type de rejet
        - ``group`` : groupe sensible de l'individu
    """
    if graph_cfg is None:
        graph_cfg = GraphConfig()
    if feature_cols is None:
        feature_cols = list(X_train_raw.columns)

    nodes: List[GraphNode] = []
    sig_to_idx: Dict[str, int] = {}
    rej = {"actionability": 0, "duplicate": 0, "implausible": 0,
           "over_budget": 0, "expanded": 0, "favorable_reached": 0}

    df0 = pd.DataFrame([x0])
    sc0 = float(predict_score(model, df0, feature_cols)[0])
    grp0 = int(get_group(df0, age_threshold)[0])
    fav0 = decision_rule.is_favorable(sc0, group=grp0, model_name=model_name)
    nnd0 = nearest_neighbor_distance(x0, X_train_raw, schema, scales)
    den0 = local_density(x0, X_train_raw, plaus_cfg.epsilon_nn, schema, scales)
    root = GraphNode(idx=0, state=x0, depth=0, parent_idx=None, action_name=None,
                     action_obj=None, cumulative_cost=0.0, score=sc0,
                     is_favorable=fav0, nn_distance=nnd0, density=den0)
    nodes.append(root)
    sig_to_idx[state_signature(x0)] = 0
    if fav0:
        return {"nodes": nodes, "favorable_paths": [[0]],
                "rejection_counts": rej, "group": grp0}

    favorable_paths: List[List[int]] = []
    frontier = [root]

    for depth in range(1, graph_cfg.max_depth + 1):
        scored_candidates = []
        for parent in frontier:
            for act in actions:
                if not act.precondition_fn(parent.state):
                    continue
                new_state = act.apply_fn(parent.state)
                if new_state is None:
                    continue
                if not is_actionable_change(x0, new_state, schema):
                    rej["actionability"] += 1
                    continue
                sig = state_signature(new_state)
                if graph_cfg.prune_duplicate_states and sig in sig_to_idx:
                    rej["duplicate"] += 1
                    continue
                nnd = nearest_neighbor_distance(new_state, X_train_raw, schema, scales)
                if graph_cfg.prune_unplausible_states and nnd > plaus_cfg.epsilon_nn:
                    rej["implausible"] += 1
                    continue
                if (graph_cfg.prune_unplausible_states
                        and plaus_cfg.joint_check_fn is not None
                        and not plaus_cfg.joint_check_fn(new_state)):
                    rej["implausible_joint"] = rej.get("implausible_joint", 0) + 1
                    continue
                new_cost = parent.cumulative_cost + act.cost_fn(parent.state)
                if (graph_cfg.max_cumulative_cost is not None
                        and new_cost > graph_cfg.max_cumulative_cost):
                    rej["over_budget"] += 1
                    continue
                den = local_density(new_state, X_train_raw, plaus_cfg.epsilon_nn,
                                     schema, scales)
                sc = float(predict_score(model, pd.DataFrame([new_state]),
                                          feature_cols)[0])
                fav = decision_rule.is_favorable(sc, group=grp0, model_name=model_name)
                node = GraphNode(
                    idx=len(nodes), state=new_state, depth=depth,
                    parent_idx=parent.idx, action_name=act.name, action_obj=act,
                    cumulative_cost=new_cost, score=sc, is_favorable=fav,
                    nn_distance=nnd, density=den,
                )
                nodes.append(node)
                sig_to_idx[sig] = node.idx
                rej["expanded"] += 1
                if fav:
                    path = []
                    cur = node
                    while cur is not None:
                        path.append(cur.idx)
                        cur = nodes[cur.parent_idx] if cur.parent_idx is not None else None
                    favorable_paths.append(list(reversed(path)))
                    rej["favorable_reached"] += 1
                else:
                    tau = decision_rule.required_threshold(group=grp0,
                                                            model_name=model_name)
                    scored_candidates.append((sc - tau, node))
                if len(nodes) >= graph_cfg.max_nodes:
                    break
            if len(nodes) >= graph_cfg.max_nodes:
                break
        scored_candidates.sort(key=lambda t: -t[0])
        frontier = [n for _, n in scored_candidates[:graph_cfg.beam_width]]
        if not frontier or len(nodes) >= graph_cfg.max_nodes:
            break

    return {"nodes": nodes, "favorable_paths": favorable_paths,
            "rejection_counts": rej, "group": grp0}


# ============================================================================
# 10. Robustesse bootstrap
# ============================================================================

@dataclass
class BootstrapConfig:
    """Configuration de l'évaluation de robustesse par bootstrap du modèle."""
    n_bootstrap: int = 10
    model_factory: Callable = None
    decision_rule: DecisionRule = None
    threshold_strategy: str = "fixed"
    rng_seed: int = 42


def fit_bootstrap_models(X_train: pd.DataFrame, y_train: np.ndarray,
                          A_train: np.ndarray,
                          config: BootstrapConfig
                          ) -> List[Tuple[Any, DecisionRule]]:
    """Entraîne B modèles sur des bootstraps du training set.

    Returns
    -------
    list of (fitted_model, decision_rule) tuples
    """
    rng = np.random.default_rng(config.rng_seed)
    n = len(X_train)
    out = []
    for b in range(config.n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        Xb = X_train.iloc[idx]
        yb = y_train[idx]
        model_b = config.model_factory()
        model_b.fit(Xb, yb)
        if config.threshold_strategy == "fixed":
            rule_b = config.decision_rule
        elif config.threshold_strategy == "recompute_global":
            tau_b = float(yb.mean())
            rule_b = GlobalThresholdRule(
                tau=tau_b, margin=getattr(config.decision_rule, "margin", 0.0))
        else:
            rule_b = config.decision_rule
        out.append((model_b, rule_b))
    return out


def bootstrap_robust_validity(
    x_state: pd.Series,
    bootstrap_models: List[Tuple[Any, DecisionRule]],
    feature_cols: Optional[List[str]] = None,
    age_threshold: float = 25.0,
    model_name: str = "boot",
) -> float:
    """Fraction des bootstraps où l'état reste favorable.

    $$\\text{RobustValidity}(\\mathbf{x}) = \\frac{1}{B} \\sum_b \\mathbb{1}\\{D(f_b(\\mathbf{x}), A) = 1\\}$$
    """
    x_df = pd.DataFrame([x_state])
    if feature_cols is None:
        feature_cols = list(x_df.columns)
    group = int(get_group(x_df, age_threshold)[0])
    n_ok = 0
    for model_b, rule_b in bootstrap_models:
        sc = float(predict_score(model_b, x_df, feature_cols)[0])
        if rule_b.is_favorable(sc, group=group, model_name=model_name):
            n_ok += 1
    return n_ok / max(len(bootstrap_models), 1)


def make_lr_pipeline_factory(cat_cols: List[str], num_cols: List[str],
                              C: float = 1e10, max_iter: int = 5000,
                              seed: int = 42) -> Callable:
    """Factory de pipelines LR non-fittés (pour les bootstraps).

    Reproduit la baseline équipe : C=1e10 (quasi-non-régularisé),
    max_iter=5000, solver lbfgs, OHE drop='first' + StandardScaler.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    def factory():
        return Pipeline([
            ("prep", ColumnTransformer([
                ("cat", OneHotEncoder(drop="first", handle_unknown="ignore",
                                      sparse_output=False), cat_cols),
                ("num", StandardScaler(), num_cols),
            ])),
            ("lr", LogisticRegression(C=C, max_iter=max_iter,
                                      solver="lbfgs", random_state=seed)),
        ])
    return factory


def make_xgb_pipeline_factory(cat_cols: List[str], num_cols: List[str],
                                n_estimators: int = 200, max_depth: int = 4,
                                learning_rate: float = 0.1,
                                seed: int = 42) -> Callable:
    """Factory de pipelines XGBoost non-fittés (pour les bootstraps)."""
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from xgboost import XGBClassifier

    def factory():
        return Pipeline([
            ("prep", ColumnTransformer([
                ("cat", OneHotEncoder(drop="first", handle_unknown="ignore",
                                      sparse_output=False), cat_cols),
                ("num", StandardScaler(), num_cols),
            ])),
            ("xgb", XGBClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                  learning_rate=learning_rate,
                                  random_state=seed, eval_metric="logloss",
                                  tree_method="hist")),
        ])
    return factory


# ============================================================================
# 11. Path, scoring multi-objectif, sélection diverse
# ============================================================================

@dataclass
class RecoursePath:
    """Représentation d'un chemin d'actions favorable.

    Attributes
    ----------
    states : list of pd.Series
        Tous les états traversés (racine + intermédiaires + final).
    actions : list of Action
        Actions appliquées entre états successifs.
    cumulative_cost : float
        Coût cumulé sur le chemin.
    final_score : float
    required_threshold : float
    final_margin : float
        score_final - threshold.
    nn_distances : list of float
        Distance NN marginale par état (incluant l'état initial).
    densities : list of int
    group : int
        Groupe sensible de l'individu initial.
    robust_validity : float, optional
        Fraction des bootstraps où l'état final reste favorable.
    """
    states: List[pd.Series]
    actions: List[Action]
    cumulative_cost: float
    final_score: float
    required_threshold: float
    final_margin: float
    nn_distances: List[float]
    densities: List[int]
    group: int
    robust_validity: Optional[float] = None
    _max_nn: float = field(default=None, init=False)
    _modified: set = field(default=None, init=False)

    @property
    def length(self) -> int:
        return len(self.actions)

    @property
    def max_nn_distance(self) -> float:
        if self._max_nn is None:
            self._max_nn = max(self.nn_distances) if self.nn_distances else 0.0
        return self._max_nn

    @property
    def mean_nn_distance(self) -> float:
        return float(np.mean(self.nn_distances)) if self.nn_distances else 0.0

    @property
    def min_density(self) -> int:
        return min(self.densities) if self.densities else 0

    @property
    def mean_density(self) -> float:
        return float(np.mean(self.densities)) if self.densities else 0.0

    @property
    def modified_features(self) -> set:
        if self._modified is None:
            self._modified = set(a.feature for a in self.actions)
        return self._modified

    @property
    def num_modified_features(self) -> int:
        return len(self.modified_features)

    @property
    def first_action(self) -> Optional[str]:
        return self.actions[0].name if self.actions else None


def graph_paths_to_path_objects(graph_result: Dict[str, Any],
                                  required_threshold: float) -> List[RecoursePath]:
    """Convertit les ``favorable_paths`` du graphe en objets ``Path`` enrichis."""
    nodes = graph_result["nodes"]
    paths = []
    for path_idxs in graph_result["favorable_paths"]:
        if not path_idxs:
            continue
        states = [nodes[i].state for i in path_idxs]
        actions = [nodes[i].action_obj for i in path_idxs[1:]]
        nn_dists = [nodes[i].nn_distance for i in path_idxs]
        dens = [nodes[i].density for i in path_idxs]
        final_node = nodes[path_idxs[-1]]
        paths.append(RecoursePath(
            states=states, actions=actions,
            cumulative_cost=final_node.cumulative_cost,
            final_score=final_node.score,
            required_threshold=required_threshold,
            final_margin=final_node.score - required_threshold,
            nn_distances=nn_dists, densities=dens,
            group=graph_result["group"],
        ))
    return paths


@dataclass
class PathScoringConfig:
    """Poids du scoring multi-objectif des chemins.

    $$\\mathrm{Obj}(P) = \\lambda_c \\tilde c + \\lambda_T \\tilde T + \\lambda_s \\tilde s
       + \\lambda_p \\widetilde{d_{NN}^{\\max}} - \\lambda_m \\tilde m - \\lambda_r \\tilde \\rho$$

    Plus bas est meilleur. ``λ_p`` porte sur la plausibilité de la **trajectoire**
    (worst state), pas seulement de l'endpoint — apport clé de PACR-AP.
    """
    lambda_cost: float = 1.0
    lambda_length: float = 0.5
    lambda_sparsity: float = 0.3
    lambda_plausibility: float = 1.0
    lambda_margin: float = 0.5
    lambda_robust: float = 0.5


def _norm(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize ; zero si variance nulle."""
    arr = np.asarray(arr, dtype=float)
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def score_paths(paths: List[RecoursePath],
                  cfg: Optional[PathScoringConfig] = None) -> np.ndarray:
    """Calcule l'objectif scalaire normalisé pour chaque chemin (lower is better)."""
    if cfg is None:
        cfg = PathScoringConfig()
    if not paths:
        return np.array([])
    cost = _norm([p.cumulative_cost for p in paths])
    length = _norm([p.length for p in paths])
    spars = _norm([p.num_modified_features for p in paths])
    plaus = _norm([p.max_nn_distance for p in paths])
    margin = _norm([p.final_margin for p in paths])
    robust = np.array([p.robust_validity if p.robust_validity is not None else 0.5
                        for p in paths])
    return (cfg.lambda_cost * cost + cfg.lambda_length * length
            + cfg.lambda_sparsity * spars + cfg.lambda_plausibility * plaus
            - cfg.lambda_margin * margin - cfg.lambda_robust * robust)


@dataclass
class SelectionConfig:
    """Configuration de la sélection top-K avec contrainte de diversité."""
    K: int = 3
    diversity_min_jaccard: float = 0.5


def _jaccard_distance(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return 1.0 - len(a & b) / max(len(a | b), 1)


def select_top_paths(paths: List[RecoursePath],
                       scoring_cfg: Optional[PathScoringConfig] = None,
                       sel_cfg: Optional[SelectionConfig] = None) -> List[RecoursePath]:
    """Sélection top-K gloutonne avec contrainte de diversité Jaccard.

    Algorithme : tri par objectif ; ajout itératif d'un chemin ssi (première
    action différente OU distance Jaccard des features modifiées
    ≥ ``diversity_min_jaccard``) vis-à-vis de tous les déjà sélectionnés.
    """
    if sel_cfg is None:
        sel_cfg = SelectionConfig()
    if not paths:
        return []
    obj = score_paths(paths, scoring_cfg)
    order = np.argsort(obj)
    selected = [paths[order[0]]]
    for i in order[1:]:
        if len(selected) >= sel_cfg.K:
            break
        cand = paths[i]
        if all((cand.first_action != s.first_action) or
               (_jaccard_distance(cand.modified_features, s.modified_features)
                >= sel_cfg.diversity_min_jaccard)
               for s in selected):
            selected.append(cand)
    return selected


# ============================================================================
# 12. API principale
# ============================================================================

def human_readable_path(p: Path, model, feature_cols: List[str],
                          epsilon_nn: float, n_bootstrap: int = 0,
                          model_name: str = "modèle") -> str:
    """Génère un texte d'explication métier pour un chemin sélectionné.

    Le texte décrit chaque étape (action + valeur avant/après + score résultant)
    et précise la plausibilité finale et la robustesse bootstrap. Disclaim
    explicite : « change la décision du modèle, ne prouve pas la causalité ».
    """
    lines = []
    sc0 = float(predict_score(model, pd.DataFrame([p.states[0]]), feature_cols)[0])
    tau = p.required_threshold
    lines.append(
        f"Sous {model_name} et la règle de décision sélectionnée, "
        f"la demande initiale reçoit une décision défavorable avec un score "
        f"de {sc0:.3f} (seuil requis : {tau:.3f}). Un chemin d'actions "
        f"plausibles vers une décision favorable est :"
    )
    cum_cost = 0.0
    for t, act in enumerate(p.actions, start=1):
        next_state = p.states[t]
        score_t = float(predict_score(model, pd.DataFrame([next_state]),
                                       feature_cols)[0])
        cum_cost += act.cost_fn(p.states[t - 1])
        favorable = score_t >= tau
        v_o, v_n = p.states[t - 1][act.feature], next_state[act.feature]
        decision_str = ("décision désormais favorable" if favorable
                        else "décision toujours défavorable")
        lines.append(
            f"  Étape {t} — {act.description}. `{act.feature}` : {v_o} → {v_n}. "
            f"Score : {score_t:.3f} ({decision_str}). Coût cumulé : {cum_cost:.3f}."
        )
    if p.robust_validity is not None:
        lines.append(
            f"Le profil final est empiriquement plausible "
            f"(distance NN max sur le chemin = {p.max_nn_distance:.3f}, "
            f"seuil ε = {epsilon_nn:.3f}) et reste favorable dans "
            f"{p.robust_validity * 100:.0f} % des modèles bootstrap (B={n_bootstrap}). "
        )
    else:
        lines.append(
            f"Le profil final est empiriquement plausible "
            f"(distance NN max sur le chemin = {p.max_nn_distance:.3f}, "
            f"seuil ε = {epsilon_nn:.3f})."
        )
    lines.append(
        "Cette explication décrit comment changer la décision du modèle sous "
        "la règle spécifiée. Elle ne prouve pas que l'individu deviendrait "
        "causalement un bon risque crédit."
    )
    return "\n".join(lines)


def render_recourse_cards(selected_paths, model, feature_cols, epsilon_nn,
                            n_bootstrap=0, all_paths=None, scoring_cfg=None,
                            model_name="modèle"):
    """Rend les K chemins sélectionnés sous forme de cartes HTML stylisées.

    Une carte par chemin, avec bordure colorée synchronisée avec la palette
    de ``plot_action_graph``. Met visuellement en évidence l'étape qui fait
    basculer la décision (badge ✓ vert). Si ``all_paths`` est fourni, le
    score multi-objectif Σ et le rang sont affichés dans l'en-tête de chaque
    carte.

    Returns
    -------
    IPython.display.HTML
        Objet auto-rendu par Jupyter à la fin d'une cellule.
    """
    from IPython.display import HTML
    palette = ["#E74C3C", "#9B59B6", "#16A085"]

    # ─── Σ + rang pour chaque chemin sélectionné (si all_paths fourni) ───
    sel_scores, sel_ranks = {}, {}
    if all_paths and len(all_paths) > 0:
        obj_all = score_paths(all_paths, scoring_cfg)
        order = np.argsort(obj_all)
        rank_of = {idx: r + 1 for r, idx in enumerate(order)}
        sigs_all = [state_signature(p.states[-1]) for p in all_paths]
        for k, sp in enumerate(selected_paths):
            sp_sig = state_signature(sp.states[-1])
            try:
                idx_in_all = sigs_all.index(sp_sig)
                sel_scores[k] = float(obj_all[idx_in_all])
                sel_ranks[k] = (rank_of[idx_in_all], len(all_paths))
            except ValueError:
                pass

    def _clean_desc(s):
        return s.replace("_", " ").capitalize()

    def _fmt_val(v):
        return f"{v:.0f}" if isinstance(v, float) else str(v)

    cards_html = []
    for k, p in enumerate(selected_paths):
        col = palette[k % len(palette)]
        sc0 = float(predict_score(model, pd.DataFrame([p.states[0]]),
                                   feature_cols)[0])
        tau = p.required_threshold

        if k in sel_scores:
            r, n = sel_ranks[k]
            meta_html = (
                f"Σ = {sel_scores[k]:+.2f} &nbsp;·&nbsp; rang {r} / {n}"
            )
        else:
            meta_html = ""

        # Étapes
        cum_cost = 0.0
        threshold_crossed = False
        step_lis = []
        for t, act in enumerate(p.actions, start=1):
            next_state = p.states[t]
            score_t = float(predict_score(model, pd.DataFrame([next_state]),
                                           feature_cols)[0])
            cum_cost += act.cost_fn(p.states[t - 1])
            v_old = _fmt_val(p.states[t - 1][act.feature])
            v_new = _fmt_val(next_state[act.feature])

            just_crossed = (not threshold_crossed) and score_t >= tau
            if score_t >= tau:
                threshold_crossed = True

            badge = ""
            score_color = "#888"
            if just_crossed:
                badge = ('&nbsp;&nbsp;<span style="color:#27AE60;'
                          'font-weight:bold;">✓ basculement</span>')
                score_color = "#27AE60"

            step_lis.append(f"""
                <li style="margin-bottom:7px;">
                  <b>{_clean_desc(act.description)}</b>
                  &nbsp;<code style="background:#EEE;padding:1.5px 6px;
                       border-radius:3px;font-size:11.5px;color:#444;">
                    {act.feature}: {v_old} → {v_new}
                  </code>
                  <br/>
                  <span style="color:{score_color};font-size:11.5px;">
                    score {score_t:.3f} · coût cumulé {cum_cost:.3f}
                  </span>
                  {badge}
                </li>
            """)

        if p.robust_validity is not None:
            footer = (
                f"Plausibilité : NN max <b>{p.max_nn_distance:.2f}</b> "
                f"(seuil ε = {epsilon_nn:.2f}) &nbsp;·&nbsp; "
                f"reste favorable dans <b>{p.robust_validity * 100:.0f} %</b> "
                f"des modèles bootstrap (B = {n_bootstrap})"
            )
        else:
            footer = (
                f"Plausibilité : NN max <b>{p.max_nn_distance:.2f}</b> "
                f"(seuil ε = {epsilon_nn:.2f})"
            )

        cards_html.append(f"""
        <div style="
            border-left:5px solid {col};
            background:#FAFAFA;
            border-radius:4px;
            padding:14px 20px;
            margin:12px 0;
            font-family:'DejaVu Sans',Arial,sans-serif;
            box-shadow:0 1px 3px rgba(0,0,0,0.07);
        ">
          <div style="display:flex;justify-content:space-between;
               align-items:baseline;margin-bottom:10px;">
            <div style="font-size:16px;font-weight:bold;color:{col};">
              Chemin #{k + 1}
            </div>
            <div style="font-size:12px;color:#666;font-family:monospace;">
              {meta_html}
            </div>
          </div>
          <div style="font-size:12.5px;color:#555;margin-bottom:12px;
               padding-bottom:8px;border-bottom:1px dashed #DDD;">
            Score initial <b>{sc0:.3f}</b> &nbsp;→&nbsp; seuil requis
            <b>{tau:.3f}</b> sous {model_name}
          </div>
          <ol style="margin:0 0 10px 0;padding-left:24px;font-size:13px;
               line-height:1.6;">
            {"".join(step_lis)}
          </ol>
          <div style="font-size:11.5px;color:#555;padding-top:8px;
               border-top:1px dashed #DDD;">
            {footer}
          </div>
        </div>
        """)

    disclaimer = (
        '<div style="font-size:11px;color:#888;font-style:italic;'
        'margin-top:8px;padding:8px 12px;background:#F5F5F5;'
        'border-radius:4px;">'
        "Ces cartes décrivent comment changer la décision du modèle sous la "
        "règle spécifiée. Elles ne prouvent pas que l'individu deviendrait "
        "causalement un bon risque crédit."
        '</div>'
    )

    return HTML("".join(cards_html) + disclaimer)


def generate_action_path_recourse(
    x0: pd.Series, model, decision_rule: DecisionRule,
    schema: Dict[str, FeatureSpec], actions: List[Action],
    X_train_raw: pd.DataFrame, plaus_cfg: PlausibilityConfig,
    scales: Dict[str, float],
    graph_cfg: Optional[GraphConfig] = None,
    boot_models: Optional[List[Tuple[Any, DecisionRule]]] = None,
    scoring_cfg: Optional[PathScoringConfig] = None,
    sel_cfg: Optional[SelectionConfig] = None,
    feature_cols: Optional[List[str]] = None,
    age_threshold: float = 25.0,
    model_name: str = "model",
    robust_min: float = 0.0,
) -> Dict[str, Any]:
    """Pipeline complet PACR-AP pour UN individu refusé.

    Étapes :
    1. Construit le graphe d'actions local.
    2. Extrait les chemins favorables.
    3. Évalue la robustesse bootstrap de chaque état final (si ``boot_models``).
    4. Filtre les chemins par robustesse minimale (si ``robust_min > 0``).
    5. Sélectionne les top-K chemins divers.

    Returns
    -------
    dict avec keys :
        - ``x0``, ``graph``, ``all_valid_paths``, ``filtered_paths``,
          ``selected_paths``, ``rejection_counts``, ``group``
    """
    if graph_cfg is None:
        graph_cfg = GraphConfig()
    if feature_cols is None:
        feature_cols = list(X_train_raw.columns)

    g = build_action_graph(x0, model, decision_rule, schema, actions,
                            X_train_raw, plaus_cfg, scales, graph_cfg,
                            feature_cols=feature_cols, age_threshold=age_threshold,
                            model_name=model_name)
    tau = decision_rule.required_threshold(group=g["group"], model_name=model_name)
    all_paths = graph_paths_to_path_objects(g, required_threshold=tau)
    if boot_models is not None and boot_models:
        for p in all_paths:
            p.robust_validity = bootstrap_robust_validity(
                p.states[-1], boot_models, feature_cols=feature_cols,
                age_threshold=age_threshold, model_name=model_name)
    valid_paths = [p for p in all_paths
                    if (p.robust_validity is None or p.robust_validity >= robust_min)]
    selected = select_top_paths(valid_paths, scoring_cfg, sel_cfg)
    return {
        "x0": x0, "graph": g,
        "all_valid_paths": all_paths,
        "filtered_paths": valid_paths,
        "selected_paths": selected,
        "rejection_counts": g["rejection_counts"],
        "group": g["group"],
    }


def generate_action_path_recourse_for_dataset(
    X_human: pd.DataFrame, A_arr: np.ndarray, model, decision_rule: DecisionRule,
    schema: Dict[str, FeatureSpec], actions: List[Action],
    X_train_raw: pd.DataFrame, plaus_cfg: PlausibilityConfig,
    scales: Dict[str, float],
    graph_cfg: Optional[GraphConfig] = None,
    boot_models: Optional[List[Tuple[Any, DecisionRule]]] = None,
    scoring_cfg: Optional[PathScoringConfig] = None,
    sel_cfg: Optional[SelectionConfig] = None,
    feature_cols: Optional[List[str]] = None,
    age_threshold: float = 25.0,
    model_name: str = "model",
    max_individuals: int = 30,
    robust_min: float = 0.0,
    verbose: bool = True,
) -> pd.DataFrame:
    """Applique PACR-AP aux refusés d'un dataset, renvoie un DataFrame résumé.

    Le DataFrame contient une ligne par refusé traité, avec les métriques du
    chemin top-1 sélectionné (ou NaN si aucun chemin trouvé). Utilisable
    directement par ``fairness_of_recourse_paths``.
    """
    if feature_cols is None:
        feature_cols = list(X_human.columns)
    scores = predict_score(model, X_human, feature_cols)
    groups = (X_human["age"].astype(float).values >= age_threshold).astype(int)
    refused = np.array([
        not decision_rule.is_favorable(s, group=g, model_name=model_name)
        for s, g in zip(scores, groups)
    ])
    refused_idx = np.where(refused)[0][:max_individuals]
    if verbose:
        print(f"Recourse pour {len(refused_idx)} refusés (sur {refused.sum()} au total).")

    records = []
    for i, idx in enumerate(refused_idx):
        if verbose and (i + 1) % 10 == 0:
            print(f"  … {i + 1}/{len(refused_idx)}")
        x = X_human.iloc[idx]
        res = generate_action_path_recourse(
            x, model, decision_rule, schema, actions, X_train_raw,
            plaus_cfg, scales, graph_cfg, boot_models, scoring_cfg, sel_cfg,
            feature_cols=feature_cols, age_threshold=age_threshold,
            model_name=model_name, robust_min=robust_min,
        )
        sel = res["selected_paths"]
        if sel:
            top = sel[0]
            records.append({
                "test_idx": int(idx), "group": int(A_arr[idx]),
                "score_orig": float(scores[idx]),
                "has_path": True, "n_selected": len(sel),
                "path_length": top.length,
                "n_modified_features": top.num_modified_features,
                "cumulative_cost": top.cumulative_cost,
                "final_score": top.final_score,
                "final_margin": top.final_margin,
                "max_nn_distance": top.max_nn_distance,
                "mean_nn_distance": top.mean_nn_distance,
                "min_density": top.min_density,
                "robust_validity": top.robust_validity,
                "first_action": top.first_action,
                "action_sequence": " → ".join(a.name for a in top.actions),
            })
        else:
            records.append({
                "test_idx": int(idx), "group": int(A_arr[idx]),
                "score_orig": float(scores[idx]),
                "has_path": False, "n_selected": 0,
                "path_length": np.nan, "n_modified_features": np.nan,
                "cumulative_cost": np.nan, "final_score": np.nan,
                "final_margin": np.nan, "max_nn_distance": np.nan,
                "mean_nn_distance": np.nan, "min_density": np.nan,
                "robust_validity": np.nan, "first_action": None,
                "action_sequence": None,
            })
    return pd.DataFrame(records)


# ============================================================================
# 13. Audit fairness *of recourse*
# ============================================================================

def fairness_of_recourse_paths(recourse_df: pd.DataFrame
                                  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calcule les métriques de recourse par groupe + les gaps jeunes − adultes.

    Métriques par groupe : ``n_refused``, ``coverage``, ``mean_length``,
    ``mean_cost``, ``median_cost``, ``mean_sparsity``, ``mean_margin``,
    ``mean_max_nn``, ``mean_mean_nn``, ``mean_robust``.

    Gaps dans le sens conventionnel : $\\Delta_M = M_{A=0} - M_{A=1}$ (jeunes − adultes).

    Returns
    -------
    per_group_df : pd.DataFrame
        Une ligne par groupe (jeunes, adultes).
    gaps_df : pd.DataFrame
        Une ligne unique avec les gaps signés.
    """
    rows = []
    for grp_val, grp_lbl in [(0, "Jeunes (A=0)"), (1, "Adultes (A=1)")]:
        sub = recourse_df[recourse_df["group"] == grp_val]
        sub_p = sub[sub["has_path"]]
        rows.append({
            "Groupe": grp_lbl, "n_refused": len(sub),
            "coverage": sub["has_path"].mean() if len(sub) > 0 else np.nan,
            "mean_length": sub_p["path_length"].mean() if len(sub_p) > 0 else np.nan,
            "mean_cost": sub_p["cumulative_cost"].mean() if len(sub_p) > 0 else np.nan,
            "median_cost": sub_p["cumulative_cost"].median() if len(sub_p) > 0 else np.nan,
            "mean_sparsity": sub_p["n_modified_features"].mean() if len(sub_p) > 0 else np.nan,
            "mean_margin": sub_p["final_margin"].mean() if len(sub_p) > 0 else np.nan,
            "mean_max_nn": sub_p["max_nn_distance"].mean() if len(sub_p) > 0 else np.nan,
            "mean_mean_nn": sub_p["mean_nn_distance"].mean() if len(sub_p) > 0 else np.nan,
            "mean_robust": sub_p["robust_validity"].mean() if len(sub_p) > 0 else np.nan,
        })
    df_grp = pd.DataFrame(rows)
    most_freq_first = {}
    for grp_val, grp_lbl in [(0, "Jeunes (A=0)"), (1, "Adultes (A=1)")]:
        sub_p = recourse_df[(recourse_df["group"] == grp_val) & recourse_df["has_path"]]
        if len(sub_p) > 0:
            ctr = Counter(sub_p["first_action"].dropna())
            top = ctr.most_common(1)
            most_freq_first[grp_lbl] = f"{top[0][0]} ({top[0][1]})" if top else ""
        else:
            most_freq_first[grp_lbl] = ""
    df_grp["first_action_top"] = df_grp["Groupe"].map(most_freq_first)
    if len(df_grp) == 2:
        j = df_grp.iloc[0]; a = df_grp.iloc[1]
        gaps = pd.DataFrame([{
            "Δ_coverage     (jeunes-ad)": j["coverage"] - a["coverage"],
            "Δ_mean_length  (jeunes-ad)": j["mean_length"] - a["mean_length"],
            "Δ_mean_cost    (jeunes-ad)": j["mean_cost"] - a["mean_cost"],
            "Δ_sparsity     (jeunes-ad)": j["mean_sparsity"] - a["mean_sparsity"],
            "Δ_margin       (jeunes-ad)": j["mean_margin"] - a["mean_margin"],
            "Δ_max_nn       (jeunes-ad)": j["mean_max_nn"] - a["mean_max_nn"],
            "Δ_robust       (jeunes-ad)": j["mean_robust"] - a["mean_robust"],
        }])
    else:
        gaps = pd.DataFrame()
    return df_grp, gaps


# ============================================================================
# 14. Plotting helpers
# ============================================================================

# Palette équipe : orange = jeunes (A=0, défavorisé), teal = adultes (A=1, favorisé)
COLOR_YOUNG = "#E67E22"
COLOR_ADULT = "#16A085"


# Labels métier en français — utilisés par tous les plots pour afficher les
# actions de manière humainement lisible (au lieu des noms internes).
FEATURE_LABEL_FR = {
    "credit_amount":   "montant",
    "duration":        "durée",
    "checking_status": "compte courant",
    "savings_status":  "épargne",
    "other_parties":   "garant",
    "employment":      "ancienneté emploi",
}


def action_short_label(action):
    """Label court et lisible pour une Action.

    Convertit les noms internes (``mined_decrease_duration_33pct``) en labels
    métier en français (``durée −33 %``). Réutilisable par tous les plots et
    par le notebook (Altair) pour éviter d'afficher les noms techniques.
    """
    name = action.name
    feat = getattr(action, "feature", "")
    label = FEATURE_LABEL_FR.get(feat, feat.replace("_", " "))
    if "decrease" in name and name.endswith("pct"):
        pct = name.rsplit("_", 1)[-1].replace("pct", "")
        return f"{label} −{pct} %"
    if "improve" in name and name.endswith("step"):
        step = name.rsplit("_", 1)[-1].replace("step", "")
        return f"{label} +{step} cran"
    if "_set_" in name:
        val = name.split(f"{feat}_", 1)[-1] if feat else name.rsplit("_", 1)[-1]
        return f"{label} → {val}"
    if name.startswith("reduce_amount"):
        return f"montant −{name.split('_')[-1].replace('pct','')} %"
    if name.startswith("reduce_duration"):
        return f"durée −{name.split('_')[-1].replace('m','')} mois"
    if name.startswith("improve_"):
        return f"{label} +1 cran"
    return name


def _add_context_badge(target, text):
    """Badge contextuel discret en haut-à-droite (modèle, seuil, etc.).

    Sépare le titre principal (purpose du plot) de la métadonnée
    contextuelle, qui reste visible mais sans saturer le titre.

    Accepte un ``Figure`` (positionnement en coords figure) ou un ``Axes``
    (positionnement en coords axes).
    """
    if not text:
        return
    import matplotlib.figure as _mfig
    style = dict(
        ha="right", va="top", fontsize=9.5, color="#444", style="italic",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F0F0F0",
                  edgecolor="#BBB", linewidth=0.8, alpha=0.92),
    )
    if isinstance(target, _mfig.Figure):
        target.text(0.985, 0.985, text, **style)
    else:
        target.text(0.99, 1.02, text, transform=target.transAxes, **style)


def plot_action_graph(graph_result, selected_paths, decision_rule, title="",
                      all_paths=None, scoring_cfg=None):
    """Graphe d'actions local + bandeau de sélection (3 vues du scoring).

    Si ``all_paths`` est fourni, le score multi-objectif Σ de chaque chemin
    sélectionné est calculé (via ``score_paths``) et affiché dans la légende
    sous la forme ``#k — Σ = X.XX (rang R/N)`` : on lit immédiatement la valeur
    scalaire qui justifie la sélection et la position du chemin parmi tous les
    candidats favorables.
    """
    import matplotlib.pyplot as plt
    nodes = graph_result["nodes"]
    grp = graph_result["group"]
    tau = decision_rule.required_threshold(group=grp)

    # ─── Calcul des scores multi-objectifs si all_paths fourni ────────────
    sel_scores = {}     # k → Σ value
    sel_ranks  = {}     # k → (rank, total)
    if all_paths is not None and len(all_paths) > 0:
        obj_all = score_paths(all_paths, scoring_cfg)
        order = np.argsort(obj_all)
        rank_of = {idx: r + 1 for r, idx in enumerate(order)}
        # Mapper chaque chemin sélectionné à son index dans all_paths via la signature finale
        sigs_all = [state_signature(p.states[-1]) for p in all_paths]
        for k, sp in enumerate(selected_paths):
            sp_sig = state_signature(sp.states[-1])
            try:
                idx_in_all = sigs_all.index(sp_sig)
                sel_scores[k] = float(obj_all[idx_in_all])
                sel_ranks[k]  = (rank_of[idx_in_all], len(all_paths))
            except ValueError:
                pass

    by_depth = {}
    for n in nodes:
        by_depth.setdefault(n.depth, []).append(n)
    max_d = max(by_depth.keys()) if by_depth else 0
    pos = {}
    max_abs_y = 0.0
    for d, nodes_d in by_depth.items():
        n_d = len(nodes_d)
        for i, n in enumerate(nodes_d):
            y = (i - (n_d - 1) / 2) / max(n_d, 1)
            pos[n.idx] = (d, y)
            if abs(y) > max_abs_y:
                max_abs_y = abs(y)

    import matplotlib.gridspec as gridspec
    has_scores = bool(sel_scores)
    if has_scores:
        fig = plt.figure(figsize=(14, 12.0))
        gs = gridspec.GridSpec(
            3, 3, height_ratios=[2.4, 1.1, 1.0], hspace=0.60, wspace=0.32,
            top=0.95, bottom=0.085, left=0.05, right=0.96,
        )
        ax = fig.add_subplot(gs[0, :])
        ax_scores = fig.add_subplot(gs[1, :])
        ax_panels = [fig.add_subplot(gs[2, k]) for k in range(3)]
    else:
        fig = plt.figure(figsize=(14, 9.5))
        gs = gridspec.GridSpec(
            2, 3, height_ratios=[2.4, 1.0], hspace=0.55, wspace=0.32,
            top=0.94, bottom=0.10, left=0.05, right=0.96,
        )
        ax = fig.add_subplot(gs[0, :])
        ax_scores = None
        ax_panels = [fig.add_subplot(gs[1, k]) for k in range(3)]
    # Marge supérieure pour que le titre n'écrase pas les labels de score
    ax.set_ylim(-max_abs_y - 0.35, max_abs_y + 0.55)

    # Arêtes de fond (toutes celles du graphe, en gris)
    for n in nodes:
        if n.parent_idx is None:
            continue
        x0, y0 = pos[n.parent_idx]; x1, y1 = pos[n.idx]
        ax.plot([x0, x1], [y0, y1], color="#BBBBBB", lw=0.6, alpha=0.5, zorder=1)

    palette = ["#E74C3C", "#9B59B6", "#16A085"]
    # Rayons d'arc différents par chemin (0 droit, ±0.22 courbés)
    arc_rads = [0.0, 0.22, -0.22]

    sig_to_idx = {state_signature(nn.state): nn.idx for nn in nodes}

    shorten = action_short_label  # helper module-level

    for k, p in enumerate(selected_paths):
        sigs = [state_signature(s) for s in p.states]
        idxs_in_g = [sig_to_idx[s] for s in sigs if s in sig_to_idx]
        col = palette[k % len(palette)]
        rad = arc_rads[k % 3]
        for step_idx, (a_i, b_i) in enumerate(zip(idxs_in_g, idxs_in_g[1:])):
            x0, y0 = pos[a_i]; x1, y1 = pos[b_i]
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                         arrowprops=dict(arrowstyle="->", lw=2.5, color=col, alpha=0.92,
                                         connectionstyle=f"arc3,rad={rad}"),
                         zorder=2)
            # Étiquette positionnée AU milieu de la courbe Bezier de l'arc.
            # matplotlib arc3,rad=R place le point de contrôle à mid + R*(dy, -dx),
            # donc le milieu de la Bezier (t=0.5) est à mid + 0.5*R*(dy, -dx).
            if step_idx < len(p.actions):
                act_short = shorten(p.actions[step_idx])
                mid_x = (x0 + x1) / 2
                mid_y = (y0 + y1) / 2
                dx, dy = x1 - x0, y1 - y0
                label_x = mid_x + 0.5 * rad * dy
                label_y = mid_y - 0.5 * rad * dx
                ax.text(label_x, label_y, act_short,
                         ha="center", va="center", fontsize=8.2, color=col,
                         fontweight="bold",
                         bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                                   edgecolor=col, linewidth=0.9, alpha=0.95),
                         zorder=5)

    # Nœuds terminaux favorables des chemins sélectionnés (étoiles colorées) :
    # on garde une trace de la COULEUR du chemin (palette synchro avec les arcs
    # et les scatters du bas) pour chaque nœud final sélectionné.
    final_node_color = {}
    for k, p in enumerate(selected_paths):
        sigs = [state_signature(s) for s in p.states]
        idxs = [sig_to_idx[s] for s in sigs if s in sig_to_idx]
        if idxs:
            final_node_color[idxs[-1]] = palette[k % len(palette)]

    # Nœuds + labels de score à 3 décimales
    for n in nodes:
        x, y = pos[n.idx]
        is_final_selected = n.idx in final_node_color
        if is_final_selected:
            color = final_node_color[n.idx]   # même couleur que l'arc du chemin
        elif n.is_favorable:
            color = "#27AE60"
        elif n.depth == 0:
            color = "#E67E22"
        else:
            color = "#3498DB"
        edge = "black" if n.depth == 0 or is_final_selected else "white"
        marker = "*" if is_final_selected else "o"
        size = 360 if is_final_selected else 200
        ax.scatter(x, y, s=size, c=color, edgecolor=edge, lw=1.6,
                    marker=marker, zorder=3)
        ax.text(x, y + 0.08, f"{n.score:.3f}", ha="center", fontsize=8.5,
                 color="black", zorder=4,
                 fontweight="bold" if is_final_selected else "normal")

    x0_root, y0_root = pos[0]
    ax.text(x0_root - 0.18, y0_root, "$x_0$", ha="right", va="center",
            fontsize=14, fontweight="bold")

    # Légende avec seuil à 3 décimales
    ax.scatter([], [], s=200, c="#E67E22", edgecolor="black", label="initial (refusé)")
    ax.scatter([], [], s=200, c="#3498DB", edgecolor="white", label="intermédiaire")
    ax.scatter([], [], s=200, c="#27AE60", edgecolor="white",
                label=f"favorable (score ≥ τ* = {tau:.3f})")
    # Entrées par chemin : ligne + étoile finale de la même couleur ; pas
    # d'entrée verte "final d'un chemin sélectionné" car ces nœuds prennent
    # désormais la couleur de leur chemin.
    for k in range(len(selected_paths)):
        col = palette[k % len(palette)]
        ax.plot([], [], color=col, lw=2.5, marker="*", markersize=14,
                markeredgecolor="black", markeredgewidth=1.2,
                label=f"chemin sélectionné #{k + 1}")

    ax.set_xlabel("Profondeur (nombre d'actions depuis $x_0$)")
    ax.set_yticks([]); ax.set_xticks(range(max_d + 1))
    ax.set_title(title or "Graphe d'actions local et chemins sélectionnés",
                  fontweight="bold", pad=14)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8.5, frameon=True)
    ax.grid(alpha=0.2)

    # ─── Bandeau bas : pourquoi ces K chemins ? (3 vues du scoring) ────────
    # Le scoring multi-objectif combine 5+ critères : on en projette 4 sur 3
    # scatters de marginales 2D. Chaque favorable du graphe = un point ; les K
    # finaux sélectionnés ressortent en étoile colorée avec leur label "#k",
    # ce qui permet à l'œil de suivre le même chemin à travers les 3 vues.
    fav_nodes = [n for n in nodes if n.is_favorable]
    nodes_by_idx = {n.idx: n for n in nodes}
    x0_state = nodes[0].state  # racine = profil initial de l'individu

    def _path_max_nn(node):
        cur, mx = node, node.nn_distance
        while cur.parent_idx is not None:
            cur = nodes_by_idx[cur.parent_idx]
            if cur.nn_distance > mx:
                mx = cur.nn_distance
        return mx

    def _n_changed(state):
        return sum(1 for f in state.index if x0_state[f] != state[f])

    # Métriques par favorable (pour le nuage gris)
    metrics_fav = []
    for n in fav_nodes:
        metrics_fav.append(dict(
            cost   = n.cumulative_cost,
            margin = n.score - tau,
            length = n.depth,
            spars  = _n_changed(n.state),
            maxnn  = _path_max_nn(n),
        ))
    # Métriques par chemin sélectionné (pour les étoiles colorées)
    metrics_sel = []
    for p in selected_paths:
        metrics_sel.append(dict(
            cost   = p.cumulative_cost,
            margin = p.final_margin,
            length = len(p.actions),
            spars  = _n_changed(p.states[-1]),
            maxnn  = max(p.nn_distances) if p.nn_distances else 0.0,
        ))

    # Spécification des 3 panneaux :
    # (xkey, ykey, xlabel, ylabel, titre, coin gagnant)
    # coin gagnant ∈ {"NW", "NE", "SW", "SE"} → où placer l'annotation "mieux"
    panel_specs = [
        ("cost",   "margin", "coût cumulé",        "marge (score − τ*)",
         "Efficacité",         "NW"),
        ("length", "spars",  "longueur (étapes)",  "sparsité (features touchées)",
         "Complexité d'action", "SW"),
        ("maxnn",  "margin", "max NN distance",    "marge (score − τ*)",
         "Plausibilité × marge", "NW"),
    ]
    corner_xy = {
        "NW": (0.04, 0.96, "left", "top",     "↖ mieux"),
        "NE": (0.96, 0.96, "right", "top",    "↗ mieux"),
        "SW": (0.04, 0.04, "left", "bottom",  "↙ mieux"),
        "SE": (0.96, 0.04, "right", "bottom", "↘ mieux"),
    }

    # Axes "entiers" pour lesquels on applique du jitter (décolle les overlaps)
    INT_KEYS = {"length", "spars"}
    _rng = np.random.default_rng(0)  # jitter déterministe → plot reproductible

    def _jitter(vals, key, scale=0.18):
        if key not in INT_KEYS:
            return np.asarray(vals, dtype=float)
        return np.asarray(vals, dtype=float) + _rng.uniform(-scale, scale, size=len(vals))

    # Tailles bulles selon multiplicité (clusters visibles sans noyer le détail)
    def _bubble_sizes(xs, ys, base=55):
        from collections import Counter
        # Bucketise les valeurs continues (round 3 chiffres) ; entiers naturels
        keys = list(zip(np.round(xs, 3), np.round(ys, 3)))
        cnt = Counter(keys)
        return [base * (1 + 0.55 * (cnt[k] - 1) ** 0.6) for k in keys]

    # ─── Panneau central : "Pourquoi ces K chemins ?" ─────────────────────
    # Courbe rang × Σ : tous les candidats favorables sont triés par Σ
    # ascendant et tracés comme une courbe monotone. La PENTE révèle si
    # beaucoup de chemins se valent (plat) ou si la sélection est tranchée
    # (raide). Les K sélectionnés ressortent en étoile colorée à leur
    # position exacte (rang, Σ) sur la courbe. Les "sauts" de rang
    # entre #1, #2, #3 sont l'empreinte visible du filtre de diversité.
    if has_scores:
        order = np.argsort(obj_all)
        sorted_obj = np.asarray(obj_all)[order]
        ranks_all = np.arange(1, len(sorted_obj) + 1)

        # Courbe monotone reliant les candidats triés par Σ
        ax_scores.plot(ranks_all, sorted_obj, color="#BBBBBB", lw=1.4,
                       alpha=0.75, zorder=1)
        ax_scores.scatter(ranks_all, sorted_obj, s=42, c="#B8B8B8",
                          edgecolor="#666", lw=0.5, alpha=0.75, zorder=2)

        # K étoiles colorées à leur position (rang, Σ)
        for k in range(len(selected_paths)):
            if k not in sel_scores:
                continue
            sigma = sel_scores[k]
            r, n = sel_ranks[k]
            col = palette[k % len(palette)]
            # Halo blanc + étoile colorée
            ax_scores.scatter([r], [sigma], s=540, c="white", edgecolor="white",
                              lw=0.1, marker="*", zorder=5)
            ax_scores.scatter([r], [sigma], s=360, c=col, edgecolor="black",
                              lw=1.5, marker="*", zorder=6)
            ax_scores.annotate(
                f"#{k + 1}   Σ = {sigma:+.2f}\nrang {r} / {n}",
                xy=(r, sigma), xycoords="data",
                xytext=(12, 10), textcoords="offset points",
                ha="left", va="bottom", fontsize=9, fontweight="bold",
                color=col,
                bbox=dict(boxstyle="round,pad=0.16", facecolor="white",
                          edgecolor=col, linewidth=0.9, alpha=0.95),
                zorder=7,
            )

        # Annotation orientatrice : "bottom-left = mieux"
        ax_scores.annotate(
            "↙ rang petit + Σ bas = meilleur compromis multi-critères",
            xy=(0.02, 0.95), xycoords="axes fraction",
            ha="left", va="top", fontsize=9, color="#444", style="italic",
        )

        ax_scores.set_xlabel(
            f"Rang parmi les {len(sorted_obj)} candidats favorables "
            f"(trié par Σ ascendant)",
            fontsize=9.5,
        )
        ax_scores.set_ylabel("Σ multi-objectif", fontsize=9.5)
        ax_scores.set_title(
            "Pourquoi ces K chemins ? — courbe de Σ trié sur tous les favorables",
            fontsize=10.5, fontweight="bold", pad=8,
        )
        ax_scores.grid(alpha=0.22)
        ax_scores.spines["top"].set_visible(False)
        ax_scores.spines["right"].set_visible(False)
        # Marge haute pour les bbox des annotations
        y_pad = (sorted_obj.max() - sorted_obj.min()) * 0.20
        ax_scores.set_ylim(sorted_obj.min() - y_pad * 0.3,
                           sorted_obj.max() + y_pad)

    for ax_p, (xk, yk, xlab, ylab, ttl, corner) in zip(ax_panels, panel_specs):
        # Nuage des favorables — jitter sur axes entiers + bubble-sizing
        if metrics_fav:
            xs_raw = [m[xk] for m in metrics_fav]
            ys_raw = [m[yk] for m in metrics_fav]
            sizes  = _bubble_sizes(xs_raw, ys_raw)
            xs_j = _jitter(xs_raw, xk)
            ys_j = _jitter(ys_raw, yk)
            ax_p.scatter(xs_j, ys_j, s=sizes, c="#B8B8B8", edgecolor="#555",
                         lw=0.5, alpha=0.55, zorder=2)
        # Trait de seuil marge=0 quand y = marge
        if yk == "margin":
            ax_p.axhline(0, color="#27AE60", lw=1.0, ls="--",
                         alpha=0.7, zorder=1)
        # Étoiles colorées des K sélectionnés — la correspondance #k se lit
        # via la palette de couleurs (synchro avec les chemins du graphe).
        for k, m in enumerate(metrics_sel):
            col = palette[k % len(palette)]
            x_sel = m[xk] + (_rng.uniform(-0.18, 0.18) if xk in INT_KEYS else 0)
            y_sel = m[yk] + (_rng.uniform(-0.18, 0.18) if yk in INT_KEYS else 0)
            # Halo blanc d'abord, étoile colorée par-dessus
            ax_p.scatter([x_sel], [y_sel], s=480, c="white", edgecolor="white",
                         lw=0.1, marker="*", zorder=5)
            ax_p.scatter([x_sel], [y_sel], s=320, c=col, edgecolor="black",
                         lw=1.5, marker="*", zorder=6)
        # Padding adaptatif : 12% de marge pour que les étoiles ne touchent jamais le bord
        all_x = [m[xk] for m in metrics_fav] + [m[xk] for m in metrics_sel]
        all_y = [m[yk] for m in metrics_fav] + [m[yk] for m in metrics_sel]
        if all_x and all_y:
            x_min, x_max = min(all_x), max(all_x)
            y_min, y_max = min(all_y), max(all_y)
            x_pad = max((x_max - x_min) * 0.12, 0.3 if xk in INT_KEYS else 0.05)
            y_pad = max((y_max - y_min) * 0.12, 0.3 if yk in INT_KEYS else 0.02)
            ax_p.set_xlim(x_min - x_pad, x_max + x_pad)
            ax_p.set_ylim(y_min - y_pad, y_max + y_pad)
        # Annotation "mieux" dans le coin gagnant
        cx, cy, ha, va, txt = corner_xy[corner]
        ax_p.annotate(txt, xy=(cx, cy), xycoords="axes fraction",
                      ha=ha, va=va, fontsize=9, color="#222",
                      fontweight="bold",
                      bbox=dict(boxstyle="round,pad=0.18", facecolor="#FFFDE7",
                                edgecolor="#BBB", linewidth=0.7, alpha=0.92))
        # Mini-stat : nombre de candidats favorables
        ax_p.text(0.98, 0.97, f"n = {len(metrics_fav)} candidats",
                  transform=ax_p.transAxes, ha="right", va="top",
                  fontsize=7.5, color="#666", style="italic")
        ax_p.set_xlabel(xlab, fontsize=9)
        ax_p.set_ylabel(ylab, fontsize=9)
        ax_p.set_title(ttl, fontsize=10.5, fontweight="bold", pad=6)
        ax_p.tick_params(labelsize=8)
        ax_p.grid(alpha=0.22)

    # Sous-titre commun sous le bandeau (légende qualitative)
    # y = 0.015 → léger gap au-dessus du bord inférieur ; fontsize bumpée à 11
    fig.text(
        0.5, 0.015,
        "★ chemins sélectionnés (mêmes couleurs que le graphe ci-dessus)   ·   "
        "● autres états favorables du graphe   ·   "
        "le coin « mieux » indique la direction préférée par le scoring",
        ha="center", va="bottom", fontsize=11, color="#333", style="italic",
    )

    plt.show()


def plot_score_progression_and_cost(res, model, feature_cols, title=""):
    """Progression du score + coût cumulé, avec zoom auto sur la zone de variation.

    L'axe Y du score est zoomé automatiquement autour de la plage observée pour
    rendre visible des variations même petites (par ex. un individu proche du seuil).
    """
    import matplotlib.pyplot as plt
    palette = ["#E74C3C", "#9B59B6", "#16A085"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
    tau_d = (res["selected_paths"][0].required_threshold
             if res["selected_paths"] else 0.5)

    # Subplot 1 : score progression
    ax = axes[0]
    all_scores = []
    for k, p in enumerate(res["selected_paths"]):
        scores = [float(predict_score(model, pd.DataFrame([s]), feature_cols)[0])
                  for s in p.states]
        all_scores.extend(scores)
        ax.plot(range(len(scores)), scores, "-o", color=palette[k % len(palette)],
                label=f"chemin #{k + 1}", lw=2.2, markersize=9)
    # Auto-zoom Y avec τ inclus
    if all_scores:
        all_with_tau = all_scores + [tau_d]
        y_min, y_max = min(all_with_tau), max(all_with_tau)
        margin = max(0.02, (y_max - y_min) * 0.25)
        ax.set_ylim(y_min - margin, y_max + margin)
    ax.axhline(tau_d, color="red", ls="--", lw=2.5, alpha=0.85)
    ax.text(0.02, tau_d, f" τ* = {tau_d:.3f}", color="red",
            fontsize=10, fontweight="bold", va="bottom",
            transform=ax.get_yaxis_transform())
    ax.set_xlabel("Étape t"); ax.set_ylabel("Score $f(x_t)$")
    ax.set_title("Progression du score")
    ax.legend(loc="best"); ax.grid(alpha=0.3)

    # Subplot 2 : coût cumulé
    ax = axes[1]
    for k, p in enumerate(res["selected_paths"]):
        costs = [0.0]; cur = 0.0
        for t, act in enumerate(p.actions):
            cur += act.cost_fn(p.states[t]); costs.append(cur)
        ax.plot(range(len(costs)), costs, "-s", color=palette[k % len(palette)],
                label=f"chemin #{k + 1}", lw=2.2, markersize=9)
    ax.set_xlabel("Étape t"); ax.set_ylabel("Coût cumulé (σ-units)")
    ax.set_title("Coût cumulé")
    ax.legend(loc="best"); ax.grid(alpha=0.3)

    plt.suptitle(title or "Trajectoires des chemins sélectionnés",
                  fontweight="bold", y=1.02)
    plt.tight_layout(); plt.show()


def plot_refused_score_distribution(scores: np.ndarray, A_test: np.ndarray,
                                     tau: float, title_prefix: str = ""):
    """Histogramme des scores **des refusés**, par groupe sensible.

    Contextualise immédiatement le déséquilibre n_jeunes vs n_adultes parmi
    les refusés : à seuil cost-optimal élevé, les scores adultes se massent
    dans la zone [0.5, τ*] alors que les jeunes sont plus polarisés.
    """
    import matplotlib.pyplot as plt
    refused_mask = scores < tau
    refused_scores = scores[refused_mask]
    refused_groups = A_test[refused_mask]
    n_J = int((refused_groups == 0).sum())
    n_A = int((refused_groups == 1).sum())

    fig, ax = plt.subplots(figsize=(11, 4.5))
    bins = np.linspace(0, tau, 26)
    ax.hist(refused_scores[refused_groups == 0], bins=bins,
            color=COLOR_YOUNG, alpha=0.78, edgecolor="white",
            label=f"Jeunes (A=0, n={n_J})")
    ax.hist(refused_scores[refused_groups == 1], bins=bins,
            color=COLOR_ADULT, alpha=0.65, edgecolor="white",
            label=f"Adultes (A=1, n={n_A})")
    ax.axvline(tau, color="red", ls="--", lw=2.2,
                label=f"τ* = {tau:.3f}")
    ax.set_xlabel("Score initial $f(x)$")
    ax.set_ylabel("Nombre d'individus refusés")
    ax.set_title("Distribution des scores chez les refusés",
                  fontsize=12.5, fontweight="bold", pad=22)
    _add_context_badge(ax, title_prefix)
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.show()


def plot_audit_panels(recourse_df: pd.DataFrame, fair_per_grp: pd.DataFrame,
                       epsilon_nn: float, title_prefix: str = ""):
    """Audit visuel 1×4 : coverage / coût / robust / plausibilité max.

    Utilise un stripplot + médiane noire au lieu de boxplot — statistiquement
    honnête sur petit n (la médiane d'un boxplot sur 3 points est trompeuse).
    L'effectif n par groupe est annoté sur les labels d'axe.
    """
    import matplotlib.pyplot as plt
    cols = [COLOR_YOUNG, COLOR_ADULT]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))

    n_J = int(fair_per_grp.iloc[0]["n_refused"])
    n_A = int(fair_per_grp.iloc[1]["n_refused"])
    # Labels sur 2 lignes : groupe (gras) + effectif (petit) ; évite le chevauchement
    xt_labels = [f"Jeunes\nn = {n_J}", f"Adultes\nn = {n_A}"]

    # (a) Coverage (bar plot — l'annotation % est utile)
    ax = axes[0]
    ax.bar(xt_labels, fair_per_grp["coverage"], color=cols,
            alpha=0.85, edgecolor="white")
    for i, v in enumerate(fair_per_grp["coverage"]):
        if pd.notna(v):
            ax.text(i, v + 0.02, f"{v:.0%}", ha="center",
                    fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.15); ax.set_ylabel("Coverage")
    ax.set_title("Coverage du recourse")
    ax.grid(axis="y", alpha=0.3)

    # (b-d) Stripplots avec médiane noire
    panel_specs = [
        ("cumulative_cost", "Coût cumulé du chemin", None, "Coût (σ-units)"),
        ("robust_validity", "Robustesse bootstrap", 0.8, "Robust validity"),
        ("max_nn_distance", "Plausibilité (NN max)", epsilon_nn, "NN distance max"),
    ]
    rng = np.random.default_rng(42)
    for ax, (col, ttl, hline, ylab) in zip(axes[1:], panel_specs):
        # On collecte les valeurs pour calculer un offset vertical de label
        # qui reste proportionnel à la plage des données (jamais clippé)
        all_vals = []
        for g in [0, 1]:
            sub = recourse_df[(recourse_df["group"] == g) & recourse_df["has_path"]][col].dropna()
            if len(sub) > 0:
                all_vals.extend(sub.tolist())
        if len(all_vals) > 1:
            y_min, y_max = min(all_vals), max(all_vals)
            y_range = max(y_max - y_min, 1e-6)
        else:
            y_min = y_max = (all_vals[0] if all_vals else 0.0)
            y_range = max(abs(y_min), 0.1)
        # Offset vertical : 4 % de la plage — assez pour décoller, pas pour clipper
        y_off = y_range * 0.04

        for i, g in enumerate([0, 1]):
            sub = recourse_df[(recourse_df["group"] == g) & recourse_df["has_path"]][col].dropna()
            if len(sub) == 0:
                continue
            x_jit = rng.normal(i, 0.06, len(sub))
            ax.scatter(x_jit, sub, color=cols[i], alpha=0.75, s=85,
                        edgecolor="white", linewidth=1.2, zorder=2)
            # Médiane : trait noir + label COMPACT centré au-dessus du trait
            # (pas de callout latéral → pas de chevauchement avec les xticks
            # ni avec le panneau voisin).
            med = sub.median()
            ax.scatter([i], [med], color="black", marker="_",
                        s=420, linewidth=3.0, zorder=3)
            ax.text(i, med + y_off, f"{med:.2f}",
                    va="bottom", ha="center",
                    fontsize=9, fontweight="bold", color="black",
                    bbox=dict(boxstyle="round,pad=0.14", facecolor="white",
                              edgecolor="black", linewidth=0.8, alpha=0.96),
                    zorder=10)
        ax.set_xticks([0, 1]); ax.set_xticklabels(xt_labels, fontsize=9)
        ax.set_xlim(-0.5, 1.5)
        # Marge verticale haute pour accueillir le bbox du label sans clipping
        ax.set_ylim(y_min - y_range * 0.08, y_max + y_range * 0.14)
        ax.set_title(ttl); ax.set_ylabel(ylab)
        ax.grid(axis="y", alpha=0.3)
        if hline is not None:
            ax.axhline(hline, color="red", ls="--", alpha=0.55,
                        label=f"seuil = {hline:.2f}")
            ax.legend(loc="lower right", fontsize=8.5)

    plt.suptitle("Audit visuel par groupe",
                  fontsize=14, fontweight="bold", y=1.03)
    _add_context_badge(fig, title_prefix)
    plt.tight_layout(); plt.show()


def _extract_features_from_sequence(action_sequence: str) -> set:
    """Parse une chaîne d'actions et retourne l'ensemble des features modifiées."""
    if not action_sequence or pd.isna(action_sequence):
        return set()
    feats = set()
    for a in action_sequence.split(" → "):
        if "credit_amount" in a or ("amount" in a and "credit" not in a):
            feats.add("credit_amount")
        elif "duration" in a:
            feats.add("duration")
        elif "checking_status" in a:
            feats.add("checking_status")
        elif "savings_status" in a:
            feats.add("savings_status")
        elif "employment" in a:
            feats.add("employment")
        elif "co_applicant" in a or "guarantor" in a or "other_parties" in a:
            feats.add("other_parties")
    return feats


def plot_feature_modification_dumbbell(recourse_df: pd.DataFrame,
                                         title_prefix: str = ""):
    """Dumbbell : pour chaque feature, % de Jeunes vs % d'Adultes qui la modifient.

    Plus parlant qu'une heatmap 2-colonnes :
    - **Longueur du segment** = ampleur du gap entre groupes (lecture immédiate)
    - **Couleur du segment** = sens du gap (orange = jeunes touchent plus, vert
      = adultes touchent plus, gris = égalité)
    - **Tri par |gap| décroissant** : les features les plus différenciantes
      remontent en haut → le lecteur voit la disparité dès la première ligne
    - **Annotation `gap = ±N pts`** dans la marge droite : la valeur exacte
      reste lisible sans avoir à calculer mentalement
    """
    import matplotlib.pyplot as plt
    sub = recourse_df[recourse_df["has_path"]].copy()
    if len(sub) == 0:
        print("Aucun chemin à visualiser.")
        return
    sub["features_modified"] = sub["action_sequence"].apply(_extract_features_from_sequence)
    all_features = sorted({f for fs in sub["features_modified"] for f in fs})
    if not all_features:
        print("Aucune feature détectée.")
        return

    # Calcul des % par feature par groupe
    n_J = int((sub["group"] == 0).sum())
    n_A = int((sub["group"] == 1).sum())
    rows = []
    for feat in all_features:
        pJ = sub[sub["group"] == 0]["features_modified"].apply(
            lambda fs: feat in fs).sum() / n_J if n_J else 0.0
        pA = sub[sub["group"] == 1]["features_modified"].apply(
            lambda fs: feat in fs).sum() / n_A if n_A else 0.0
        rows.append({"feature": feat, "pct_J": pJ, "pct_A": pA, "gap": pJ - pA})
    df_feat = pd.DataFrame(rows)
    # Tri par |gap| croissant → matplotlib affiche bottom-up donc le plus grand
    # |gap| arrive en haut (lecture top-down)
    df_feat["abs_gap"] = df_feat["gap"].abs()
    df_feat = df_feat.sort_values("abs_gap", ascending=True).reset_index(drop=True)

    n_feat = len(df_feat)
    fig, ax = plt.subplots(figsize=(11.5, max(4.5, 0.65 * n_feat + 1.8)))

    # Labels métier en français (cohérence avec plot_action_graph)
    FR = {"credit_amount": "montant du crédit", "duration": "durée",
          "checking_status": "compte courant", "savings_status": "épargne",
          "other_parties": "garant / co-emprunteur", "employment": "ancienneté emploi"}

    for i, row in df_feat.iterrows():
        y = i
        pJ, pA = row["pct_J"], row["pct_A"]
        gap = row["gap"]
        # Couleur du segment selon sens du gap
        if abs(gap) < 0.03:
            seg_color, seg_alpha = "#999999", 0.45
        elif gap > 0:  # Jeunes touchent plus
            seg_color, seg_alpha = COLOR_YOUNG, 0.6
        else:           # Adultes touchent plus
            seg_color, seg_alpha = COLOR_ADULT, 0.6
        # Segment épais reliant les 2 points
        ax.plot([pJ, pA], [y, y], color=seg_color, lw=5.5, alpha=seg_alpha,
                solid_capstyle="round", zorder=1)
        # Points colorés par groupe
        ax.scatter([pJ], [y], color=COLOR_YOUNG, s=280, edgecolor="white",
                   linewidth=2.0, zorder=3)
        ax.scatter([pA], [y], color=COLOR_ADULT, s=280, edgecolor="white",
                   linewidth=2.0, zorder=3)
        # % près de chaque point (à l'extérieur du segment)
        left_x, right_x = min(pJ, pA), max(pJ, pA)
        left_pct, right_pct = (pJ, pA) if pJ <= pA else (pA, pJ)
        ax.text(left_x - 0.02, y, f"{left_pct:.0%}", va="center", ha="right",
                fontsize=10, fontweight="bold", color="#222", zorder=4)
        ax.text(right_x + 0.02, y, f"{right_pct:.0%}", va="center", ha="left",
                fontsize=10, fontweight="bold", color="#222", zorder=4)
        # Annotation gap dans la marge droite
        gap_pts = gap * 100
        gap_col = (COLOR_YOUNG if gap > 0.03 else
                   COLOR_ADULT if gap < -0.03 else "#888")
        ax.text(1.18, y, f"{gap_pts:+.0f} pts",
                va="center", ha="right", fontsize=10.5, fontweight="bold",
                color=gap_col, zorder=4)

    # Référence visuelle à 50 %
    ax.axvline(0.5, color="#DDD", ls=":", lw=1, zorder=0)

    # En-tête de la colonne "gap"
    ax.text(1.18, n_feat - 0.35, "gap (J − A)",
            va="bottom", ha="right", fontsize=9.5, fontweight="bold",
            color="#444", style="italic")

    # Y-ticks = labels métier
    ax.set_yticks(range(n_feat))
    ax.set_yticklabels([FR.get(f, f) for f in df_feat["feature"]], fontsize=11)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0 %", "25 %", "50 %", "75 %", "100 %"])
    ax.set_xlim(-0.08, 1.20)
    ax.set_ylim(-0.6, n_feat - 0.4 + 0.6)
    ax.set_xlabel("% d'individus du groupe modifiant la feature (chemin top-1)")

    # Légende explicite via éléments fantômes (matched colors)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_YOUNG,
               markersize=12, label=f"Jeunes (n = {n_J})"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_ADULT,
               markersize=12, label=f"Adultes (n = {n_A})"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10,
              frameon=True, ncol=2)

    ax.set_title(
        "Quelles features chaque groupe modifie-t-il ?\n"
        "trié par |gap| décroissant · segment = ampleur de la disparité",
        fontweight="bold", fontsize=12.5, pad=22,
    )
    _add_context_badge(ax, title_prefix)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.show()


def plot_barbell_before_after(recourse_df: pd.DataFrame, tau: float,
                                title_prefix: str = ""):
    """Vue 'avant → après' par individu : segments horizontaux ordonnés par score initial.

    Lecture immédiate : qui gagne beaucoup, qui pousse juste, qui n'a aucun chemin.
    Permet de voir d'un coup d'œil la disparité par groupe.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    df = recourse_df.copy().sort_values("score_orig").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(11, max(5, 0.32 * len(df))))

    for i, row in df.iterrows():
        color = COLOR_YOUNG if row["group"] == 0 else COLOR_ADULT
        x0 = row["score_orig"]
        if row["has_path"]:
            xf = row["final_score"]
            ax.plot([x0, xf], [i, i], color=color, lw=2.2, alpha=0.55, zorder=1)
            ax.scatter([x0], [i], color=color, s=70, marker="o",
                        edgecolor="white", linewidth=1.2, zorder=2)
            ax.scatter([xf], [i], color=color, s=120, marker=">",
                        edgecolor="black", linewidth=1.0, zorder=3)
        else:
            ax.scatter([x0], [i], color=color, s=110, marker="x",
                        linewidths=2.8, zorder=2)

    ax.axvline(tau, color="red", ls="--", lw=2.2, alpha=0.8)
    ax.text(tau, len(df) + 0.5, f"τ* = {tau:.3f}", color="red",
            fontsize=11, fontweight="bold", ha="center")

    ax.set_xlabel("Score $f(x)$")
    ax.set_ylabel("Individus refusés, triés par score initial ↑")
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_title("Avant → après recourse par individu — " + title_prefix + "\n"
                  "•  =  score initial    │    ▶  =  score final    │    ×  =  aucun chemin trouvé",
                  fontweight="bold")

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_YOUNG,
                markersize=11, label="Jeune (A=0)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_ADULT,
                markersize=11, label="Adulte (A=1)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout(); plt.show()


def plot_gap_confidence_intervals(recourse_df: pd.DataFrame,
                                    n_bootstrap: int = 500, seed: int = 42,
                                    title_prefix: str = "") -> pd.DataFrame:
    """Bootstrap CI 95% sur les gaps de fairness — réponse rigoureuse au n imbalance.

    Re-échantillonne ``n_bootstrap`` fois avec remise les individus du recourse_df,
    recalcule les gaps à chaque resample, puis affiche médiane + IC 95%. Permet
    de juger si les gaps observés sont distinguables du bruit.
    """
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(seed)
    n = len(recourse_df)
    boot = {"Δ_coverage (J−A)": [], "Δ_cost (J−A)": [], "Δ_robust (J−A)": []}

    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        sub = recourse_df.iloc[idx]
        J = sub[sub["group"] == 0]; A = sub[sub["group"] == 1]
        if len(J) == 0 or len(A) == 0:
            continue
        cov_J = J["has_path"].mean(); cov_A = A["has_path"].mean()
        Jp = J[J["has_path"]]; Ap = A[A["has_path"]]
        if len(Jp) == 0 or len(Ap) == 0:
            continue
        boot["Δ_coverage (J−A)"].append(cov_J - cov_A)
        boot["Δ_cost (J−A)"].append(Jp["cumulative_cost"].mean() - Ap["cumulative_cost"].mean())
        boot["Δ_robust (J−A)"].append(Jp["robust_validity"].mean() - Ap["robust_validity"].mean())

    fig, ax = plt.subplots(figsize=(11, 4))
    metrics = list(boot.keys())
    summary_rows = []

    for i, m in enumerate(metrics):
        gaps = np.array([g for g in boot[m] if not np.isnan(g)])
        if len(gaps) == 0:
            continue
        median = float(np.median(gaps))
        ci_lo = float(np.percentile(gaps, 2.5))
        ci_hi = float(np.percentile(gaps, 97.5))
        crosses_zero = (ci_lo < 0 < ci_hi)
        # Couleur : gris si CI traverse 0, sinon teal (en faveur jeunes si pos) ou orange (sinon)
        if crosses_zero:
            color = "#888"
        else:
            color = COLOR_ADULT if median > 0 else COLOR_YOUNG

        # Densité bootstrap (violon horizontal)
        ax.violinplot([gaps], positions=[i], vert=False, widths=0.6,
                       showmeans=False, showmedians=False, showextrema=False)
        # IC 95%
        ax.plot([ci_lo, ci_hi], [i, i], color=color, lw=4, alpha=0.85)
        # Médiane
        ax.plot([median], [i], "o", color=color, markersize=15, zorder=3,
                 markeredgecolor="white", markeredgewidth=2)
        # Annotation
        signif = "" if crosses_zero else " ✓ significatif"
        ax.text(max(ci_hi, 0) + 0.05, i,
                 f"{median:+.3f}  [{ci_lo:+.3f}, {ci_hi:+.3f}]{signif}",
                 va="center", fontsize=10.5,
                 fontweight="bold" if not crosses_zero else "normal")
        summary_rows.append({"metric": m, "median": median,
                              "ci_2.5": ci_lo, "ci_97.5": ci_hi,
                              "significant": not crosses_zero})

    # Ligne rouge de référence à 0 : un IC qui la croise = gap non significatif
    ax.axvline(0, color="#E74C3C", lw=2.0, alpha=0.85, zorder=1)
    ax.set_yticks(range(len(metrics))); ax.set_yticklabels(metrics)
    ax.set_xlabel("Gap (jeunes − adultes)")
    ax.set_title(
        f"Intervalles de confiance bootstrap des gaps de fairness "
        f"(95 %, B = {n_bootstrap})\n"
        f"✓ significatif = l'intervalle ne traverse pas 0",
        fontsize=12.5, fontweight="bold", pad=22,
    )
    _add_context_badge(ax, title_prefix)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout(); plt.show()
    return pd.DataFrame(summary_rows)


def plot_sensitivity_gaps(sens_df, title="", context=""):
    """Dot plot horizontal compact : 3 métriques × 3 variants sur un même axe.

    Les valeurs sont annotées en offset (en points, indépendant des unités data)
    pour ne jamais chevaucher les labels d'axe Y ni se faire clipper aux bords.
    """
    import matplotlib.pyplot as plt
    metrics = ["Δ_cov", "Δ_cost", "Δ_robust"]
    metric_labels = ["Δ coverage", "Δ mean cost", "Δ robust validity"]
    variants = sens_df["variant"].tolist()
    variant_y_offset = {v: (i - 1) * 0.22 for i, v in enumerate(variants)}
    variant_colors = ["#8366B6", "#4FA197", "#5AAC68"]
    variant_markers = ["o", "s", "^"]

    # Calcul de la plage globale pour fixer xlim AVANT de placer les annotations
    all_vals = []
    for v in variants:
        for m in metrics:
            val = sens_df.loc[sens_df["variant"] == v, m].iloc[0]
            if not pd.isna(val):
                all_vals.append(float(val))
    if not all_vals:
        return
    v_min = min(min(all_vals), 0.0)
    v_max = max(max(all_vals), 0.0)
    v_range = max(v_max - v_min, 0.01)
    # Marge gauche+droite (18 %) : zoom serré sur les données mais
    # assez d'espace pour que les labels en offset (±10 pt) ne clippent jamais.
    pad = v_range * 0.18

    fig, ax = plt.subplots(figsize=(11.5, 4.4))
    ax.set_xlim(v_min - pad, v_max + pad)

    for v_idx, variant in enumerate(variants):
        for m_idx, metric in enumerate(metrics):
            val = sens_df.loc[sens_df["variant"] == variant, metric].iloc[0]
            if pd.isna(val):
                continue
            y = m_idx + variant_y_offset[variant]
            ax.plot([0, val], [y, y], color=variant_colors[v_idx], lw=2.2, alpha=0.55)
            ax.scatter([val], [y], color=variant_colors[v_idx],
                        marker=variant_markers[v_idx], s=170,
                        edgecolor="white", linewidth=1.5, zorder=3,
                        label=variant if m_idx == 0 else None)
            # Offset en POINTS (insensible aux unités data) → 9pt de chaque côté
            # du marker, va="center" pour rester aligné sur le point.
            # Cas spécial : si |val| est très petit (≤ 5 % de la plage),
            # forcer le label vers la droite quoi qu'il arrive, pour rester
            # dans la zone marge et ne JAMAIS empiéter sur les y-ticklabels.
            if abs(val) < v_range * 0.05:
                dx_pts, ha = 10, "left"
            elif val >= 0:
                dx_pts, ha = 10, "left"
            else:
                dx_pts, ha = -10, "right"
            ax.annotate(
                f"{val:+.3f}", xy=(val, y), xycoords="data",
                xytext=(dx_pts, 0), textcoords="offset points",
                va="center", ha=ha,
                fontsize=9.5, fontweight="bold", color=variant_colors[v_idx],
                zorder=4,
            )

    ax.axvline(0, color="black", lw=1.3, alpha=0.75)
    ax.set_yticks(range(len(metrics))); ax.set_yticklabels(metric_labels)
    ax.set_xlabel("Gap (jeunes − adultes)")
    ax.set_title(title or "Sensibilité du verdict au schéma d'actions",
                  fontsize=12.5, fontweight="bold", pad=22)
    _add_context_badge(ax, context)
    # Légende placée HORS de la zone des annotations pour éviter tout conflit
    ax.legend(title="Variant", loc="upper left",
              bbox_to_anchor=(1.01, 1.0), fontsize=10, frameon=True)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout(); plt.show()


def plot_joint_plausibility_quadrant(X_train, schema, scales, epsilon_nn,
                                       rho_lof, joint_score_fn,
                                       n_sample: int = 300, seed: int = 42):
    """Scatter NN distance vs LOF score avec zones de rejet ombrées.

    Les zones "rejetées par marginal" et "rejetées par joint" sont colorées en rouge
    transparent. Les points captés par le filtre joint mais pas le marginal sont mis
    en évidence (plus gros, contour noir) car ce sont les plus pédagogiquement
    intéressants — c'est ce que l'extension §17 apporte.
    """
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_train), size=min(n_sample, len(X_train)), replace=False)
    rows = []
    for i in idx:
        x = X_train.iloc[int(i)]
        d_all = mixed_distance(x, X_train, schema, scales)
        nnd = float(np.sort(d_all)[1])
        jsc = joint_score_fn(x)
        rows.append({"nn_dist": nnd, "lof_score": jsc,
                      "marginal_pass": nnd <= epsilon_nn,
                      "joint_pass": jsc >= rho_lof})
    viz_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9.5, 6))

    # Zones de rejet ombrées
    x_min, x_max = viz_df["nn_dist"].min() * 0.95, viz_df["nn_dist"].max() * 1.05
    y_min, y_max = viz_df["lof_score"].min() * 1.05, viz_df["lof_score"].max() * 0.95
    ax.axvspan(epsilon_nn, x_max, color="red", alpha=0.06, zorder=0)
    ax.axhspan(y_min, rho_lof, color="red", alpha=0.06, zorder=0)

    # Points par catégorie
    groups = [
        ("Plausible (les deux filtres)",  viz_df.marginal_pass & viz_df.joint_pass,
         "#16A085", 0.6, 50, 0.8),
        ("Marginal OK, joint KO",         viz_df.marginal_pass & ~viz_df.joint_pass,
         "#E67E22", 0.95, 140, 1.8),
        ("Joint OK, marginal KO",         ~viz_df.marginal_pass & viz_df.joint_pass,
         "#9B59B6", 0.85, 70, 1.2),
        ("Ni l'un ni l'autre",            ~viz_df.marginal_pass & ~viz_df.joint_pass,
         "#E74C3C", 0.85, 70, 1.2),
    ]
    for lbl, mask, color, alpha, size, lw in groups:
        sub = viz_df[mask]
        edge = "black" if "joint KO" in lbl else "white"
        ax.scatter(sub["nn_dist"], sub["lof_score"], c=color, alpha=alpha,
                    s=size, edgecolor=edge, linewidth=lw,
                    label=f"{lbl} (n={len(sub)})", zorder=3 if "joint KO" in lbl else 2)

    ax.axvline(epsilon_nn, color="red", ls="--", lw=1.5, alpha=0.7)
    ax.text(epsilon_nn, y_max, f" ε = {epsilon_nn:.2f}", color="red",
            fontsize=10, fontweight="bold", va="bottom")
    ax.axhline(rho_lof, color="blue", ls="--", lw=1.5, alpha=0.7)
    ax.text(x_min, rho_lof, f" ρ_LOF = {rho_lof:.3f}", color="blue",
            fontsize=10, fontweight="bold", ha="left", va="bottom")

    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Distance NN marginale (← plus proche)")
    ax.set_ylabel("Score LOF joint (↑ plus inlier)")
    ax.set_title("Plausibilité marginale vs jointe sur le training set\n"
                  "Zones rouges = rejet | points en gras = captés par filtre joint seulement",
                  fontweight="bold", fontsize=11)
    ax.legend(loc="lower right", fontsize=9.5)
    ax.grid(alpha=0.25)
    plt.tight_layout(); plt.show()
    return viz_df
