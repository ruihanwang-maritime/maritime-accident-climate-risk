# %% [markdown]
# M3 reanalysis: marine weather and accident consequences
#
# This script is organised as VS Code/Jupyter cells. Run it from top to bottom.
# Primary analyses:
#   1. Data validation and a common complete-case sample.
#   2. Separate wave-height and wind-speed dose-response models, each adjusted
#      for vessel age, vessel size and cargo subtype.
#   3. Separate-hazard age effect-modification models, controlling for size.
#   4. Separate-hazard size effect-modification models, controlling for age.
#   5. Interaction-contrast tables and a complete 2 x 2 Figure 3.
# Optional analyses:
#   6. Mutually adjusted wave-and-wind sensitivity models.
#   7. Full age-and-size interaction models fitted separately by hazard.
#   8. Nonlinear age/size specifications.
#   9. Binomial GLM sensitivity analysis with the same specification.

# %% 0. Imports and user settings
from __future__ import annotations
import os
DATA_ROOT = os.environ.get('WRH_DATA_ROOT', 'G:/WRH_data')  # local data folder; see README

from pathlib import Path
import json
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import norm

import statsmodels.api as sm
import statsmodels.formula.api as smf


# ----------------------------- USER SETTINGS -----------------------------
# Change only this path before the first run.
DEFAULT_ACC_FP = (
    rf"{DATA_ROOT}/AIS_processing/accident_data/"
    r"accident_all_weather_2011_2021_age.parquet"
)
# The environment override is useful for testing without editing this file:
#   Windows cmd: set M3_ACC_FP=C:\path\data.parquet
#   PowerShell:  $env:M3_ACC_FP = "C:\path\data.parquet"
ACC_FP = Path(os.environ.get("M3_ACC_FP", DEFAULT_ACC_FP))

# Fixed-effect specification. Choose exactly one:
#   "area_month" : casualty region x calendar month + year FE
#   "g10"        : 10-degree grid + calendar month + year FE
#   "g10_month"  : 10-degree grid x calendar month + year FE
FE_MODE = "area_month"

# Keep True when a reasonably populated cargo_subtype variable can be built.
USE_CARGO_SUBTYPE_CONTROL = True

# Optional analyses. First test the script with these False; enable afterwards.
RUN_FULL_INTERACTION = True
RUN_NONLINEAR = False
RUN_GLM_SENSITIVITY = False

# Output location.
SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
OUT_DIR = SCRIPT_DIR / "M3_reanalysis_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Main variable names.
CLUSTER_COL = "g5"
OUTCOMES = {
    "ser": "Serious outcome",
    "tloss": "Total loss",
}

# Total loss combines actual and constructive total losses.
# Missing loss_type, war loss and all other records are coded as zero.
TOTAL_LOSS_CATEGORIES = {
    "actual total loss",
    "constructive total loss",
}

WAVE_EDGES = [0.0, 1.25, 2.5, 4.0, 6.0, np.inf]
WAVE_LABELS = ["<1.25", "1.25-2.5", "2.5-4", "4-6", ">=6"]
WIND_EDGES = [0.0, 8.0, 10.8, 13.9, 17.2, np.inf]
WIND_LABELS = ["<8", "8-10.8", "10.8-13.9", "13.9-17.2", ">=17.2"]

HAZARDS = {
    "wv": {
        "name": "Significant wave height",
        "symbol": r"$H_s$ (m)",
        "reference": WAVE_LABELS[0],
        "labels": WAVE_LABELS,
        "colour_light": "#8fb9d1",
        "colour_dark": "#1a4a63",
    },
    "wd": {
        "name": "10-m wind speed",
        "symbol": r"$U_{10}$ (m s$^{-1}$)",
        "reference": WIND_LABELS[0],
        "labels": WIND_LABELS,
        "colour_light": "#e9a58c",
        "colour_dark": "#93402a",
    },
}

MARKERS = {"ser": "s", "tloss": "o"}
DODGE = {"ser": -0.10, "tloss": 0.10}
LN2 = np.log(2.0)

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 80)


# %% 1. Data validation and sample construction
CARGO_INCLUDE_REGEX = (
    r"bulk|cellular|container|general cargo|roll on roll off|ro-ro|roro|reefer|"
    r"vehicle carrier|livestock|wood-chip|barge carrier|ore carrier|aggregates|"
    r"combined|cargo ship"
)
CARGO_EXCLUDE_REGEX = (
    r"passenger|ferry|cruise|tanker|fishing|trawler|tug|supply|offshore|"
    r"dredg|yacht|patrol"
)


def normalise_text(series: pd.Series) -> pd.Series:
    """Return a stripped, lower-case pandas StringDtype series."""
    return series.astype("string").str.strip().str.lower()


def classify_cargo_subtype(vessel_type: pd.Series) -> pd.Series:
    """Create a small set of cargo subtypes for covariate adjustment."""
    vt = normalise_text(vessel_type)
    conditions = [
        vt.str.contains(r"bulk|ore carrier|wood-chip|aggregates", regex=True, na=False),
        vt.str.contains(r"container|cellular", regex=True, na=False),
        vt.str.contains(r"roll on roll off|ro-ro|roro|vehicle carrier", regex=True, na=False),
        vt.str.contains(r"reefer|refrigerated", regex=True, na=False),
        vt.str.contains(r"general cargo|cargo ship|combined|barge carrier", regex=True, na=False),
    ]
    choices = ["bulk", "container", "ro_ro_vehicle", "reefer", "general_other"]
    return pd.Series(
        np.select(conditions, choices, default="other_cargo"),
        index=vessel_type.index,
        dtype="string",
    )


def validate_required_columns(df: pd.DataFrame) -> None:
    required = {
        "vessel_type",
        "swh_daily_max",
        "wind_daily_max",
        "age",
        "GT",
        "serious",
        "loss_type",
        "casualty_location",
        "casualty_date",
        "lat",
        "lon",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def build_analysis_sample(path: Path) -> tuple[pd.DataFrame, dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Input parquet not found: {path}\n"
            "Edit ACC_FP in Section 0 before running the script."
        )

    raw = pd.read_parquet(path)
    validate_required_columns(raw)
    audit: dict[str, object] = {"n_raw": int(len(raw))}

    print("\nRaw serious values:")
    print(raw["serious"].value_counts(dropna=False).head(30))
    print("\nRaw loss_type values:")
    print(raw["loss_type"].value_counts(dropna=False).head(50))

    vt = normalise_text(raw["vessel_type"])
    is_cargo = (
        vt.str.contains(CARGO_INCLUDE_REGEX, regex=True, na=False)
        & ~vt.str.contains(CARGO_EXCLUDE_REGEX, regex=True, na=False)
    )
    df = raw.loc[is_cargo].copy()
    audit["n_cargo_regex"] = int(len(df))

    # Robust serious-outcome mapping. Unknown non-missing values remain missing
    # and are reported rather than silently being coded as zero.
    serious_norm = normalise_text(df["serious"])
    serious_map = {
        "true": 1,
        "1": 1,
        "1.0": 1,
        "yes": 1,
        "y": 1,
        "false": 0,
        "0": 0,
        "0.0": 0,
        "no": 0,
        "n": 0,
    }
    df["ser"] = serious_norm.map(serious_map).astype("Float64")
    unknown_serious = serious_norm[df["ser"].isna() & serious_norm.notna()]
    if len(unknown_serious):
        print("\nWARNING: unmapped serious values; these rows will be excluded:")
        print(unknown_serious.value_counts(dropna=False))
    audit["n_unknown_or_missing_serious"] = int(df["ser"].isna().sum())

    # Total loss combines actual and constructive total loss. In this database,
    # loss_type is populated for loss records and is otherwise commonly None;
    # therefore None and all other categories are coded as zero.
    loss_norm = normalise_text(df["loss_type"])
    df["tloss"] = (
        loss_norm
        .isin(TOTAL_LOSS_CATEGORIES)
        .fillna(False)
        .astype("int8")
    )
    audit["n_loss_type_none"] = int(loss_norm.isna().sum())
    audit["n_actual_total_loss_raw_cargo"] = int(
        loss_norm.eq("actual total loss").fillna(False).sum()
    )
    audit["n_constructive_total_loss_raw_cargo"] = int(
        loss_norm.eq("constructive total loss").fillna(False).sum()
    )
    audit["n_combined_total_loss_raw_cargo"] = int(df["tloss"].sum())

    # Parse dates and numeric variables explicitly.
    df["date"] = pd.to_datetime(df["casualty_date"], errors="coerce")
    for col in ["swh_daily_max", "wind_daily_max", "age", "GT", "lat", "lon"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    valid_core = (
        df["ser"].notna()
        & df["tloss"].notna()
        & df["date"].notna()
        & df["swh_daily_max"].ge(0)
        & df["wind_daily_max"].ge(0)
        & df["age"].between(0, 60, inclusive="both")
        & df["GT"].gt(0)
        & df["lat"].between(-90, 90, inclusive="both")
        & df["lon"].between(-180, 180, inclusive="both")
        & df["casualty_location"].notna()
    )
    audit["n_invalid_core"] = int((~valid_core).sum())
    df = df.loc[valid_core].copy()

    df["ser"] = df["ser"].astype(int)
    df["tloss"] = df["tloss"].astype(int)

    # Weather categories. right=False means [lower, upper).
    df["wv"] = pd.cut(
        df["swh_daily_max"], WAVE_EDGES, right=False,
        labels=WAVE_LABELS, ordered=True,
    )
    df["wd"] = pd.cut(
        df["wind_daily_max"], WIND_EDGES, right=False,
        labels=WIND_LABELS, ordered=True,
    )
    if df[["wv", "wd"]].isna().any().any():
        raise ValueError("Weather binning produced missing categories.")

    # Time and spatial identifiers.
    df["yr"] = df["date"].dt.year.astype(int).astype(str)
    df["mo"] = df["date"].dt.month.astype(int).astype(str)
    df["area"] = normalise_text(df["casualty_location"])
    df["area_month"] = df["area"] + "_m" + df["mo"]

    lat5 = np.floor(df["lat"] / 5.0).astype(int) * 5
    lon5 = np.floor(df["lon"] / 5.0).astype(int) * 5
    lat10 = np.floor(df["lat"] / 10.0).astype(int) * 10
    lon10 = np.floor(df["lon"] / 10.0).astype(int) * 10
    df["g5"] = lat5.astype(str) + "_" + lon5.astype(str)
    df["g10"] = lat10.astype(str) + "_" + lon10.astype(str)
    df["g10_month"] = df["g10"] + "_m" + df["mo"]

    df["cargo_subtype"] = classify_cargo_subtype(df["vessel_type"])

    # Centre continuous modifiers at the common-sample medians.
    age_reference = float(df["age"].median())
    gt_reference = float(df["GT"].median())
    df["age5c"] = (df["age"] - age_reference) / 5.0
    df["lgtc"] = np.log(df["GT"] / gt_reference)
    df["age5c_sq"] = df["age5c"] ** 2
    df["lgtc_sq"] = df["lgtc"] ** 2

    # Compatibility with older Patsy versions. Pandas StringDtype columns such
    # as string[python] can trigger
    #   TypeError: Cannot interpret 'string[python]' as a data type
    # when Patsy builds C(...) terms. Plain object columns are handled reliably.
    patsy_string_columns = [
        "yr",
        "mo",
        "area",
        "area_month",
        "g5",
        "g10",
        "g10_month",
        "cargo_subtype",
    ]
    for column in patsy_string_columns:
        df[column] = df[column].astype(object)

    audit.update(
        {
            "n_analysis": int(len(df)),
            "age_reference_years": age_reference,
            "gt_reference": gt_reference,
            "n_clusters_g5": int(df["g5"].nunique()),
            "n_g10": int(df["g10"].nunique()),
            "n_areas": int(df["area"].nunique()),
        }
    )

    print("\nSample audit:")
    print(json.dumps(audit, indent=2))
    print("\nOutcome prevalence:")
    for outcome, label in OUTCOMES.items():
        print(f"  {label}: {df[outcome].sum():,}/{len(df):,} ({100*df[outcome].mean():.2f}%)")
    print("\nWave-category counts:")
    print(df["wv"].value_counts(sort=False))
    print("\nWind-category counts:")
    print(df["wd"].value_counts(sort=False))
    print("\nCargo-subtype counts:")
    print(df["cargo_subtype"].value_counts())

    return df, audit


g, SAMPLE_AUDIT = build_analysis_sample(ACC_FP)


# %% 2. Fixed effects, model helpers and statistical contrasts
def fixed_effect_rhs(mode: str) -> str:
    choices = {
        "area_month": "C(area_month) + C(yr)",
        "g10": "C(g10) + C(mo) + C(yr)",
        "g10_month": "C(g10_month) + C(yr)",
    }
    if mode not in choices:
        raise ValueError(f"FE_MODE must be one of {list(choices)}; got {mode!r}")
    return choices[mode]


FE_RHS = fixed_effect_rhs(FE_MODE)
SUBTYPE_RHS = " + C(cargo_subtype)" if USE_CARGO_SUBTYPE_CONTROL else ""


def cterm(var: str, reference: str) -> str:
    """Patsy treatment-coded categorical term with an explicit reference."""
    return f'C({var}, Treatment(reference="{reference}"))'


WV = cterm("wv", WAVE_LABELS[0])
WD = cterm("wd", WIND_LABELS[0])


def fit_lpm(df: pd.DataFrame, outcome: str, rhs: str):
    formula = f"{outcome} ~ {rhs} + {FE_RHS}"
    model = smf.ols(formula, data=df, missing="raise")
    result = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": df[CLUSTER_COL], "use_correction": True},
    )
    result._m3_formula = formula
    return result


def fit_binomial_glm(df: pd.DataFrame, outcome: str, rhs: str):
    """Optional sensitivity model. No automatic fixed-effect fallback."""
    formula = f"{outcome} ~ {rhs} + {FE_RHS}"
    model = smf.glm(
        formula,
        data=df,
        family=sm.families.Binomial(),
        missing="raise",
    )
    result = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": df[CLUSTER_COL], "use_correction": True},
        maxiter=200,
    )
    result._m3_formula = formula
    return result


def category_token(label: str) -> str:
    return f"[T.{label}]"


def find_main_term(result, variable: str, label: str) -> str:
    candidates = [
        name for name in result.params.index
        if name.startswith(f"C({variable}")
        and category_token(label) in name
        and ":" not in name
    ]
    if len(candidates) != 1:
        raise KeyError(
            f"Expected one main term for {variable}={label}; found {candidates}"
        )
    return candidates[0]


def find_interaction_term(result, variable: str, label: str, modifier: str) -> str:
    candidates = [
        name for name in result.params.index
        if name.startswith(f"C({variable}")
        and category_token(label) in name
        and modifier in name.split(":")
        and ":" in name
    ]
    if len(candidates) != 1:
        raise KeyError(
            f"Expected one interaction for {variable}={label} x {modifier}; "
            f"found {candidates}"
        )
    return candidates[0]


def interaction_terms(result, variable: str, modifier: str) -> list[str]:
    return [
        name for name in result.params.index
        if name.startswith(f"C({variable}")
        and modifier in name.split(":")
        and ":" in name
    ]


def main_category_terms(result, variable: str) -> list[str]:
    return [
        name for name in result.params.index
        if name.startswith(f"C({variable}") and ":" not in name
    ]


def joint_pvalue(result, terms: list[str]) -> float:
    if not terms:
        return np.nan
    names = list(result.params.index)
    R = np.zeros((len(terms), len(names)))
    for row, term in enumerate(terms):
        R[row, names.index(term)] = 1.0
    return float(np.asarray(result.f_test(R).pvalue).squeeze())


def linear_contrast(result, weights: dict[str, float], scale: float = 1.0) -> dict:
    names = list(result.params.index)
    vector = np.zeros(len(names))
    for term, weight in weights.items():
        vector[names.index(term)] = weight

    beta = np.asarray(result.params)
    covariance = np.asarray(result.cov_params())
    estimate = float(vector @ beta) * scale
    variance = float(vector @ covariance @ vector) * (scale ** 2)
    se = np.sqrt(max(variance, 0.0))
    z = estimate / se if se > 0 else np.nan
    p = float(2.0 * norm.sf(abs(z))) if np.isfinite(z) else np.nan
    return {
        "b": estimate,
        "se": se,
        "lo": estimate - 1.96 * se,
        "hi": estimate + 1.96 * se,
        "p": p,
    }


def p_text(p: float) -> str:
    if not np.isfinite(p):
        return "NA"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


print("\nPrimary fixed effects:", FE_RHS)
print("Cluster variable:", CLUSTER_COL, f"({g[CLUSTER_COL].nunique()} clusters)")


# %% 3. Primary separate-hazard dose-response models
# These models estimate the overall association with each operational weather
# indicator. Naturally co-occurring marine conditions are part of that total
# association; the other hazard is not held constant in the primary model.
main_models = {}
MAIN_RHS = {}
for outcome in OUTCOMES:
    for hazard, meta in HAZARDS.items():
        hazard_term = cterm(hazard, meta["reference"])
        rhs = f"{hazard_term} + age5c + lgtc{SUBTYPE_RHS}"
        MAIN_RHS[hazard] = rhs
        main_models[(outcome, hazard)] = fit_lpm(g, outcome, rhs)


def build_main_effect_table(
    models: dict,
    jointly_adjusted: bool = False,
) -> pd.DataFrame:
    rows = []
    for outcome, outcome_label in OUTCOMES.items():
        for hazard, meta in HAZARDS.items():
            result = models[outcome] if jointly_adjusted else models[(outcome, hazard)]
            pj = joint_pvalue(result, main_category_terms(result, hazard))
            for index, label in enumerate(meta["labels"]):
                if index == 0:
                    effect = {"b": 0.0, "se": 0.0, "lo": 0.0, "hi": 0.0, "p": np.nan}
                else:
                    term = find_main_term(result, hazard, label)
                    effect = linear_contrast(result, {term: 1.0}, scale=100.0)
                rows.append(
                    {
                        "outcome": outcome,
                        "outcome_label": outcome_label,
                        "hazard": hazard,
                        "category": label,
                        "category_index": index,
                        **effect,
                        "joint_p": pj,
                        "n": int(result.nobs),
                    }
                )
    return pd.DataFrame(rows)


T_MAIN = build_main_effect_table(main_models)
print("\nPrimary separate-hazard dose-response estimates (percentage points):")
print(T_MAIN.to_string(index=False))
T_MAIN.to_csv(OUT_DIR / "table_main_separate_hazards.csv", index=False)


# %% 4. Mutually adjusted wind-and-wave sensitivity models
# These coefficients have a different estimand: the wind association at a
# fixed wave category, and vice versa. They are not used in the main figure.
JOINT_RHS = f"{WV} + {WD} + age5c + lgtc{SUBTYPE_RHS}"
joint_models = {
    outcome: fit_lpm(g, outcome, JOINT_RHS)
    for outcome in OUTCOMES
}
T_JOINT = build_main_effect_table(joint_models, jointly_adjusted=True)
print("\nSensitivity: mutually adjusted wave-and-wind estimates (percentage points):")
print(T_JOINT.to_string(index=False))
T_JOINT.to_csv(OUT_DIR / "table_sensitivity_joint_weather.csv", index=False)


def build_highest_category_comparison() -> pd.DataFrame:
    rows = []
    for outcome, outcome_label in OUTCOMES.items():
        for hazard, meta in HAZARDS.items():
            category = meta["labels"][-1]
            for specification, result in [
                ("separate_hazard_adjusted", main_models[(outcome, hazard)]),
                ("mutually_adjusted", joint_models[outcome]),
            ]:
                term = find_main_term(result, hazard, category)
                effect = linear_contrast(result, {term: 1.0}, scale=100.0)
                rows.append(
                    {
                        "outcome": outcome,
                        "outcome_label": outcome_label,
                        "hazard": hazard,
                        "category": category,
                        "specification": specification,
                        **effect,
                        "n": int(result.nobs),
                    }
                )
    return pd.DataFrame(rows)


T_HIGH_COMPARISON = build_highest_category_comparison()
print("\nHighest-category comparison across model specifications:")
print(T_HIGH_COMPARISON.to_string(index=False))
T_HIGH_COMPARISON.to_csv(
    OUT_DIR / "table_highest_category_model_comparison.csv", index=False
)


# %% 5. Primary separate-hazard age effect-modification models
age_models = {}
AGE_RHS = {}
for outcome in OUTCOMES:
    for hazard, meta in HAZARDS.items():
        hazard_term = cterm(hazard, meta["reference"])
        rhs = f"{hazard_term} * age5c + lgtc{SUBTYPE_RHS}"
        AGE_RHS[hazard] = rhs
        age_models[(outcome, hazard)] = fit_lpm(g, outcome, rhs)


def build_modifier_table(
    models: dict,
    modifier: str,
    scale: float,
) -> pd.DataFrame:
    """
    Return both:
      slope_*       = modifier slope within each weather category;
      interaction_* = slope difference relative to the reference category.
    For age, scale=100 gives pp per five-year increase because age5c is in five-year units.
    For size, scale=100*ln(2) gives pp per doubling of GT.
    """
    rows = []
    for outcome, outcome_label in OUTCOMES.items():
        for hazard, meta in HAZARDS.items():
            result = models[(outcome, hazard)]
            if modifier not in result.params.index:
                raise KeyError(f"Main modifier term {modifier!r} not found in model.")
            interaction_names = interaction_terms(result, hazard, modifier)
            pj = joint_pvalue(result, interaction_names)

            for index, label in enumerate(meta["labels"]):
                slope_weights = {modifier: 1.0}
                if index == 0:
                    interaction_effect = {
                        "b": 0.0, "se": 0.0, "lo": 0.0, "hi": 0.0, "p": np.nan
                    }
                else:
                    interaction_name = find_interaction_term(
                        result, hazard, label, modifier
                    )
                    slope_weights[interaction_name] = 1.0
                    interaction_effect = linear_contrast(
                        result, {interaction_name: 1.0}, scale=scale
                    )

                slope_effect = linear_contrast(result, slope_weights, scale=scale)
                rows.append(
                    {
                        "outcome": outcome,
                        "outcome_label": outcome_label,
                        "hazard": hazard,
                        "category": label,
                        "category_index": index,
                        "slope_b": slope_effect["b"],
                        "slope_se": slope_effect["se"],
                        "slope_lo": slope_effect["lo"],
                        "slope_hi": slope_effect["hi"],
                        "slope_p": slope_effect["p"],
                        "interaction_b": interaction_effect["b"],
                        "interaction_se": interaction_effect["se"],
                        "interaction_lo": interaction_effect["lo"],
                        "interaction_hi": interaction_effect["hi"],
                        "interaction_p": interaction_effect["p"],
                        "joint_interaction_p": pj,
                        "n": int(result.nobs),
                    }
                )
    return pd.DataFrame(rows)


T_AGE = build_modifier_table(age_models, modifier="age5c", scale=100.0)
print("\nAge gradients and interaction contrasts (pp per five years):")
print(T_AGE.to_string(index=False))
T_AGE.to_csv(OUT_DIR / "table_age_interactions.csv", index=False)


# %% 6. Primary separate-hazard size effect-modification models
size_models = {}
SIZE_RHS = {}
for outcome in OUTCOMES:
    for hazard, meta in HAZARDS.items():
        hazard_term = cterm(hazard, meta["reference"])
        rhs = f"{hazard_term} * lgtc + age5c{SUBTYPE_RHS}"
        SIZE_RHS[hazard] = rhs
        size_models[(outcome, hazard)] = fit_lpm(g, outcome, rhs)

T_SIZE = build_modifier_table(size_models, modifier="lgtc", scale=100.0 * LN2)
print("\nSize gradients and interaction contrasts (pp per doubling of GT):")
print(T_SIZE.to_string(index=False))
T_SIZE.to_csv(OUT_DIR / "table_size_interactions.csv", index=False)


# %% 7. Weather contrasts at representative ages and vessel sizes
def build_weather_contrasts_at_values(
    models: dict,
    modifier: str,
    values: list[tuple[str, float, float]],
) -> pd.DataFrame:
    """
    Calculate the adjusted weather-category contrast at selected modifier values.

    values contains tuples of:
      (display label, original-scale value, centred model-scale value)

    For example, the weather contrast at age a is:
      weather main coefficient + (weather x age interaction) * age5c(a).
    """
    rows = []
    for outcome, outcome_label in OUTCOMES.items():
        for hazard, meta in HAZARDS.items():
            result = models[(outcome, hazard)]
            for category_index, category in enumerate(meta["labels"]):
                for value_label, original_value, centred_value in values:
                    if category_index == 0:
                        effect = {
                            "b": 0.0,
                            "se": 0.0,
                            "lo": 0.0,
                            "hi": 0.0,
                            "p": np.nan,
                        }
                    else:
                        main_name = find_main_term(result, hazard, category)
                        interaction_name = find_interaction_term(
                            result, hazard, category, modifier
                        )
                        effect = linear_contrast(
                            result,
                            {
                                main_name: 1.0,
                                interaction_name: centred_value,
                            },
                            scale=100.0,
                        )
                    rows.append(
                        {
                            "outcome": outcome,
                            "outcome_label": outcome_label,
                            "hazard": hazard,
                            "category": category,
                            "category_index": category_index,
                            "modifier": modifier,
                            "value_label": value_label,
                            "original_value": original_value,
                            "centred_model_value": centred_value,
                            **effect,
                            "n": int(result.nobs),
                        }
                    )
    return pd.DataFrame(rows)


age_reference = float(SAMPLE_AUDIT["age_reference_years"])
age_display_values = [5.0, 15.0, 25.0, 35.0]
age_values = [
    (f"Age {age:.0f} years", age, (age - age_reference) / 5.0)
    for age in age_display_values
]
T_WEATHER_BY_AGE = build_weather_contrasts_at_values(
    age_models,
    modifier="age5c",
    values=age_values,
)
T_WEATHER_BY_AGE.to_csv(
    OUT_DIR / "table_weather_effects_at_representative_ages.csv",
    index=False,
)

gt_reference = float(SAMPLE_AUDIT["gt_reference"])
gt_quantiles = g["GT"].quantile([0.10, 0.50, 0.90])
gt_values = [
    (
        f"GT p{int(q*100):02d}",
        float(gt_value),
        float(np.log(gt_value / gt_reference)),
    )
    for q, gt_value in gt_quantiles.items()
]
T_WEATHER_BY_SIZE = build_weather_contrasts_at_values(
    size_models,
    modifier="lgtc",
    values=gt_values,
)
T_WEATHER_BY_SIZE.to_csv(
    OUT_DIR / "table_weather_effects_at_representative_GT.csv",
    index=False,
)

print("\nWeather contrasts at representative ages:")
print(T_WEATHER_BY_AGE.to_string(index=False))
print("\nWeather contrasts at representative GT values:")
print(T_WEATHER_BY_SIZE.to_string(index=False))


# %% 8. Full age-and-size interaction models fitted separately by hazard
full_models = {}
full_test_rows = []
if RUN_FULL_INTERACTION:
    FULL_RHS = {}
    for outcome, outcome_label in OUTCOMES.items():
        for hazard, meta in HAZARDS.items():
            hazard_term = cterm(hazard, meta["reference"])
            rhs = (
                f"{hazard_term} * age5c + {hazard_term} * lgtc"
                f"{SUBTYPE_RHS}"
            )
            FULL_RHS[hazard] = rhs
            result = fit_lpm(g, outcome, rhs)
            full_models[(outcome, hazard)] = result
            for modifier in ["age5c", "lgtc"]:
                terms = interaction_terms(result, hazard, modifier)
                full_test_rows.append(
                    {
                        "outcome": outcome,
                        "outcome_label": outcome_label,
                        "hazard": hazard,
                        "modifier": modifier,
                        "joint_interaction_p": joint_pvalue(result, terms),
                        "n_terms": len(terms),
                        "n": int(result.nobs),
                    }
                )

    T_FULL_TESTS = pd.DataFrame(full_test_rows)
    print("\nFull-model interaction tests:")
    print(T_FULL_TESTS.to_string(index=False))
    T_FULL_TESTS.to_csv(OUT_DIR / "table_full_model_interaction_tests.csv", index=False)


# %% 9. Nonlinear age and size specifications (optional robustness tests)
nonlinear_rows = []
if RUN_NONLINEAR:
    for outcome, outcome_label in OUTCOMES.items():
        for hazard, meta in HAZARDS.items():
            hazard_term = cterm(hazard, meta["reference"])

            # Nonlinear age interactions; size remains a control.
            age_nl_rhs = (
                f"{hazard_term} * (age5c + age5c_sq)"
                f" + lgtc{SUBTYPE_RHS}"
            )
            age_nl = fit_lpm(g, outcome, age_nl_rhs)
            terms = (
                interaction_terms(age_nl, hazard, "age5c")
                + interaction_terms(age_nl, hazard, "age5c_sq")
            )
            nonlinear_rows.append(
                {
                    "outcome": outcome,
                    "outcome_label": outcome_label,
                    "hazard": hazard,
                    "modifier": "age",
                    "specification": "linear_plus_quadratic",
                    "joint_interaction_p": joint_pvalue(age_nl, terms),
                    "n_terms": len(terms),
                }
            )

            # Nonlinear log(GT) interactions; age remains a control.
            size_nl_rhs = (
                f"{hazard_term} * (lgtc + lgtc_sq)"
                f" + age5c{SUBTYPE_RHS}"
            )
            size_nl = fit_lpm(g, outcome, size_nl_rhs)
            terms = (
                interaction_terms(size_nl, hazard, "lgtc")
                + interaction_terms(size_nl, hazard, "lgtc_sq")
            )
            nonlinear_rows.append(
                {
                    "outcome": outcome,
                    "outcome_label": outcome_label,
                    "hazard": hazard,
                    "modifier": "log_GT",
                    "specification": "linear_plus_quadratic",
                    "joint_interaction_p": joint_pvalue(size_nl, terms),
                    "n_terms": len(terms),
                }
            )

    T_NONLINEAR = pd.DataFrame(nonlinear_rows)
    print("\nNonlinear interaction robustness tests:")
    print(T_NONLINEAR.to_string(index=False))
    T_NONLINEAR.to_csv(OUT_DIR / "table_nonlinear_interaction_tests.csv", index=False)


# %% 10. Binomial GLM sensitivity analysis (optional; same FE, no fallback)
glm_rows = []
if RUN_GLM_SENSITIVITY:
    for outcome, outcome_label in OUTCOMES.items():
        for hazard in HAZARDS:
            try:
                result = fit_binomial_glm(g, outcome, MAIN_RHS[hazard])
                terms = main_category_terms(result, hazard)
                glm_rows.append(
                    {
                        "outcome": outcome,
                        "outcome_label": outcome_label,
                        "hazard": hazard,
                        "joint_p": joint_pvalue(result, terms),
                        "converged": bool(getattr(result, "converged", True)),
                        "n": int(result.nobs),
                    }
                )
            except Exception as exc:
                warnings.warn(
                    f"Binomial GLM failed for {outcome}, {hazard} under the "
                    f"unchanged FE specification: {exc}"
                )
                glm_rows.append(
                    {
                        "outcome": outcome,
                        "outcome_label": outcome_label,
                        "hazard": hazard,
                        "joint_p": np.nan,
                        "converged": False,
                        "n": len(g),
                    }
                )

    T_GLM = pd.DataFrame(glm_rows)
    print("\nBinomial GLM sensitivity tests:")
    print(T_GLM.to_string(index=False))
    T_GLM.to_csv(OUT_DIR / "table_glm_sensitivity.csv", index=False)


# %% 11. Revised Figure 3 generated completely in Python
INK = "#1a1a1a"
GREY = "#72777d"
LIGHT_GREY = "#f3f4f5"
SPINE = "#c8cdd2"


def colour_for(hazard: str, outcome: str) -> str:
    return (
        HAZARDS[hazard]["colour_light"]
        if outcome == "ser"
        else HAZARDS[hazard]["colour_dark"]
    )


def add_alternating_vertical_bands(ax, n: int) -> None:
    for index in range(0, n, 2):
        ax.axvspan(index - 0.5, index + 0.5, color=LIGHT_GREY, lw=0, zorder=0)


def style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SPINE)
    ax.spines["bottom"].set_color(SPINE)
    ax.tick_params(colors=GREY, labelsize=8.5)
    ax.set_axisbelow(True)


def plot_dose_response(ax, hazard: str, panel_label: str) -> None:
    meta = HAZARDS[hazard]
    labels = meta["labels"]
    x = np.arange(len(labels))
    add_alternating_vertical_bands(ax, len(labels))

    joint_lines = []
    for outcome, outcome_label in OUTCOMES.items():
        data = (
            T_MAIN[(T_MAIN["hazard"] == hazard) & (T_MAIN["outcome"] == outcome)]
            .sort_values("category_index")
        )
        colour = colour_for(hazard, outcome)
        xx = x + DODGE[outcome]
        ax.plot(xx, data["b"], color=colour, lw=1.4, zorder=2)
        # The reference category is represented once by a shared diamond below.
        ax.errorbar(
            xx[1:],
            data["b"].iloc[1:],
            yerr=1.96 * data["se"].iloc[1:],
            fmt=MARKERS[outcome],
            ms=5.3,
            mfc=colour,
            mec=colour,
            ecolor=colour,
            elinewidth=1.1,
            capsize=2.5,
            zorder=3,
        )
        for xi, row in zip(xx[1:], data.iloc[1:].itertuples()):
            offset = 0.7 if outcome == "ser" else -0.9
            ax.text(
                xi,
                row.b + offset,
                f"{row.b:.2f}",
                ha="center",
                va="bottom" if outcome == "ser" else "top",
                fontsize=7.2,
                color=colour,
            )
        joint_lines.append(f"{outcome_label}: $P_{{joint}}$ {p_text(data['joint_p'].iloc[0])}")

    # One shared reference diamond at zero.
    ax.plot(0, 0, marker="D", ms=5.2, mfc="white", mec=GREY, mew=1.2, zorder=5)
    ax.axhline(0, color=GREY, lw=0.8, ls=(0, (4, 3)))
    ax.set_xticks(x)
    display_labels = [label.replace(">=", "≥") for label in labels]
    ax.set_xticklabels(display_labels, fontsize=8.4)
    ax.set_xlim(-0.55, len(labels) - 0.45)
    ax.set_xlabel(meta["symbol"], fontsize=9.2, color=INK)
    ax.set_ylabel("Adjusted probability difference (pp)", fontsize=9.2, color=INK)
    ax.grid(axis="y", color="#e5e8ea", lw=0.5)
    ax.text(
        0.02,
        0.97,
        "\n".join(joint_lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
        color=INK,
    )
    ax.text(
        -0.12,
        1.08,
        panel_label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        color=INK,
    )
    ax.set_title(meta["name"], fontsize=10.2, loc="left", pad=8, color=INK)
    style_axis(ax)


def finite_limits(series_low: pd.Series, series_high: pd.Series) -> tuple[float, float]:
    low = float(np.nanmin(np.r_[series_low.to_numpy(), [0.0]]))
    high = float(np.nanmax(np.r_[series_high.to_numpy(), [0.0]]))
    span = max(high - low, 1.0)
    return low - 0.12 * span, high + 0.12 * span


AGE_XLIM = finite_limits(T_AGE["interaction_lo"], T_AGE["interaction_hi"])
SIZE_XLIM = finite_limits(T_SIZE["interaction_lo"], T_SIZE["interaction_hi"])


def plot_interaction_contrasts(
    ax,
    table: pd.DataFrame,
    hazard: str,
    modifier: str,
    xlim: tuple[float, float],
    show_ylabels: bool,
    panel_label: str | None = None,
) -> None:
    meta = HAZARDS[hazard]
    labels = meta["labels"]
    n = len(labels)
    y = np.arange(n)

    for index in range(n):
        if index % 2 == 0:
            ax.axhspan(index - 0.5, index + 0.5, color=LIGHT_GREY, lw=0, zorder=0)

    joint_lines = []
    for outcome, outcome_label in OUTCOMES.items():
        data = (
            table[(table["hazard"] == hazard) & (table["outcome"] == outcome)]
            .sort_values("category_index")
        )
        colour = colour_for(hazard, outcome)
        yy = y + DODGE[outcome]
        for ypos, row in zip(yy, data.itertuples()):
            if row.category_index == 0:
                ax.plot(
                    0,
                    ypos,
                    marker=MARKERS[outcome],
                    ms=5.0,
                    mfc="white",
                    mec=colour,
                    mew=1.1,
                    zorder=3,
                )
                continue

            significant = np.isfinite(row.interaction_p) and row.interaction_p < 0.05
            ax.plot(
                [row.interaction_lo, row.interaction_hi],
                [ypos, ypos],
                color=colour,
                lw=1.1,
                alpha=0.78,
                zorder=1,
            )
            ax.plot(
                row.interaction_b,
                ypos,
                marker=MARKERS[outcome],
                ms=5.0,
                mfc=colour if significant else "white",
                mec=colour,
                mew=1.1,
                zorder=3,
            )
        joint_lines.append(
            f"{outcome_label}: $P_{{interaction}}$ "
            f"{p_text(data['joint_interaction_p'].iloc[0])}"
        )

    ax.axvline(0, color=GREY, lw=0.8, ls=(0, (4, 3)))
    ax.set_yticks(y)
    if show_ylabels:
        ax.set_yticklabels(
            [label.replace(">=", "≥") for label in labels],
            fontsize=8.2,
        )
    else:
        ax.set_yticklabels([])
    ax.invert_yaxis()
    ax.set_xlim(*xlim)
    ax.grid(axis="x", color="#e5e8ea", lw=0.5)
    ax.text(
        0.50,
        1.015,
        "   ".join(joint_lines),
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=6.6,
        color=INK,
    )

    if modifier == "age":
        ax.set_title("Age interaction", fontsize=9.2, color=INK, pad=25)
        ax.set_xlabel(
            "Difference in age gradient\n(pp per 5 years vs reference)",
            fontsize=8.3,
            color=INK,
        )
    else:
        ax.set_title("Size interaction", fontsize=9.2, color=INK, pad=25)
        ax.set_xlabel(
            "Difference in size gradient\n(pp per GT doubling vs reference)",
            fontsize=8.3,
            color=INK,
        )

    if panel_label:
        ax.text(
            -0.22,
            1.10,
            panel_label,
            transform=ax.transAxes,
            fontsize=13,
            fontweight="bold",
            color=INK,
        )
    style_axis(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


DOSE_LEGEND_HANDLES = [
    Line2D(
        [], [], marker="s", ls="", ms=5.5, mfc="#6f8794", mec="#6f8794",
        label="Serious outcome",
    ),
    Line2D(
        [], [], marker="o", ls="", ms=5.5, mfc="#344c59", mec="#344c59",
        label="Total loss",
    ),
    Line2D(
        [], [], marker="D", ls="", ms=5.0, mfc="white", mec=GREY,
        label="Reference",
    ),
]

INTERACTION_LEGEND_HANDLES = DOSE_LEGEND_HANDLES[:2] + [
    Line2D(
        [], [], marker="o", ls="", ms=5.0, mfc="white", mec=GREY,
        label="Interaction 95% CI includes 0",
    ),
]


def save_figure_pair(fig, stem: str) -> list[Path]:
    """Save one standalone panel as vector PDF and high-resolution PNG."""
    pdf_path = OUT_DIR / f"{stem}.pdf"
    png_path = OUT_DIR / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=500, bbox_inches="tight")
    plt.close(fig)
    return [pdf_path, png_path]


def make_dose_response_figure(hazard: str, panel_label: str, stem: str) -> list[Path]:
    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=300)
    plot_dose_response(ax, hazard, panel_label)
    fig.legend(
        handles=DOSE_LEGEND_HANDLES,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=3,
        frameon=False,
        fontsize=8.0,
    )
    fig.subplots_adjust(left=0.14, right=0.98, top=0.88, bottom=0.22)
    return save_figure_pair(fig, stem)


def make_interaction_figure(
    table: pd.DataFrame,
    hazard: str,
    modifier: str,
    xlim: tuple[float, float],
    panel_label: str,
    stem: str,
) -> list[Path]:
    fig, ax = plt.subplots(figsize=(5.4, 4.8), dpi=300)
    plot_interaction_contrasts(
        ax,
        table,
        hazard,
        modifier,
        xlim,
        show_ylabels=True,
        panel_label=panel_label,
    )
    fig.legend(
        handles=INTERACTION_LEGEND_HANDLES,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=3,
        frameon=False,
        fontsize=7.4,
    )
    fig.subplots_adjust(left=0.20, right=0.97, top=0.84, bottom=0.24)
    return save_figure_pair(fig, stem)


# Generate every component independently. No combined multi-panel figure is made.
FIGURE_FILES = []
FIGURE_FILES += make_dose_response_figure("wv", "a", "figure3a_wave_dose_response")
FIGURE_FILES += make_interaction_figure(
    T_AGE, "wv", "age", AGE_XLIM, "b1", "figure3b1_wave_age_interaction"
)
FIGURE_FILES += make_interaction_figure(
    T_SIZE, "wv", "size", SIZE_XLIM, "b2", "figure3b2_wave_size_interaction"
)
FIGURE_FILES += make_dose_response_figure("wd", "c", "figure3c_wind_dose_response")
FIGURE_FILES += make_interaction_figure(
    T_AGE, "wd", "age", AGE_XLIM, "d1", "figure3d1_wind_age_interaction"
)
FIGURE_FILES += make_interaction_figure(
    T_SIZE, "wd", "size", SIZE_XLIM, "d2", "figure3d2_wind_size_interaction"
)


# %% 12. Save model metadata and final checks
model_metadata = {
    "input_file": str(ACC_FP),
    "output_directory": str(OUT_DIR),
    "fixed_effect_mode": FE_MODE,
    "fixed_effect_formula": FE_RHS,
    "cluster_column": CLUSTER_COL,
    "use_cargo_subtype_control": USE_CARGO_SUBTYPE_CONTROL,
    "sample_audit": SAMPLE_AUDIT,
    "primary_weather_specification": "wave_and_wind_modelled_separately",
    "main_rhs": MAIN_RHS,
    "joint_sensitivity_rhs": JOINT_RHS,
    "age_rhs": AGE_RHS,
    "size_rhs": SIZE_RHS,
    "run_full_interaction": RUN_FULL_INTERACTION,
    "run_nonlinear": RUN_NONLINEAR,
    "run_glm_sensitivity": RUN_GLM_SENSITIVITY,
    "total_loss_categories": sorted(TOTAL_LOSS_CATEGORIES),
}

with open(OUT_DIR / "model_metadata.json", "w", encoding="utf-8") as stream:
    json.dump(model_metadata, stream, indent=2, ensure_ascii=False)

expected_files = [
    OUT_DIR / "table_main_separate_hazards.csv",
    OUT_DIR / "table_sensitivity_joint_weather.csv",
    OUT_DIR / "table_highest_category_model_comparison.csv",
    OUT_DIR / "table_age_interactions.csv",
    OUT_DIR / "table_size_interactions.csv",
    OUT_DIR / "table_weather_effects_at_representative_ages.csv",
    OUT_DIR / "table_weather_effects_at_representative_GT.csv",
    OUT_DIR / "model_metadata.json",
    *FIGURE_FILES,
]
missing_outputs = [str(path) for path in expected_files if not path.exists()]
if missing_outputs:
    raise RuntimeError(f"Expected outputs were not created: {missing_outputs}")

print("\nAnalysis completed successfully.")
print("Outputs saved to:", OUT_DIR)
for path in expected_files:
    print(" ", path.name)
