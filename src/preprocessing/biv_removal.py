"""
biv_removal.py
==============
Decides which BIV-flagged rows to remove and WHY.  Nothing is deleted: it only adds columns,
so you filter on them and can always undo the decision.

Run after growth_scores.add_growth_scores():

    from growth_scores import load_refs, add_growth_scores
    from biv_removal import add_biv_decision, biv_summary, export_review

    out = add_growth_scores(df, refs, sex_col="sexo", birth_col="data_nascimento",
                            visit_col="atendimento", height_col="altura",
                            weight_col="peso", date_format="%Y-%m-%d")
    out = add_biv_decision(out)
    biv_summary(out)
    export_review(out, "biv_to_review.csv")      # rows you must judge by hand
    clean = out[~out["biv_removed"]]             # analysis dataset

New columns
-----------
    biv_any        True if any BIV flag (low or high) is set
    biv_reason     why the row was flagged for removal / review (see rules below), else ""
    biv_action     'remove'  -> automatically removed (rules 1-3)
                   'review'  -> BIV but no automatic rule applies: judge by hand
                   'keep'    -> no BIV flag
    biv_removed    True when biv_action == 'remove'   (or also 'review', see remove_review)

Rules, applied in this order (a row gets the FIRST rule it matches)
-------------------------------------------------------------------
    1. age_suspect      the file's own idade_suspeita flag is True
    2. age_vs_size      scored with WHO (age < 2 years) but height >= `tall_cm`:
                        a baby cannot be that tall, so the age (dates) is wrong
    3. contradictory    one measure flagged implausibly LOW and another implausibly HIGH
                        for the same child: a swapped or mistyped value
    4. (no rule)        any other BIV flag -> 'review'

Rules 1-3 point at a wrong AGE or a typo, which would also corrupt the blood-pressure lookup.
Rule 4 can be a real severely obese or very small child (the clinic is cardiology), so it is
not removed automatically.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_biv_decision(df: pd.DataFrame, *, suspect_col: str = "idade_suspeita",
                     height_col: str = "altura", tall_cm: float = 100.0,
                     remove_review: bool = False) -> pd.DataFrame:
    """Add biv_any, biv_reason, biv_action, biv_removed.  `remove_review=True` also removes the
    'review' rows (use it for a sensitivity analysis, not as the default)."""
    out = df.copy()
    low = out["biv_low"].eq(1)
    high = out["biv_high"].eq(1)
    any_biv = low | high

    suspect = out[suspect_col].fillna(False).astype(bool) if suspect_col in out else pd.Series(False, index=out.index)
    tall_baby = out["growth_ref"].eq("WHO") & (pd.to_numeric(out[height_col], errors="coerce") >= tall_cm)
    contradictory = low & high

    reason = pd.Series("", index=out.index, dtype="object")
    # lowest priority first, so higher-priority rules overwrite
    reason[any_biv] = "other_biv"
    reason[any_biv & contradictory] = "contradictory"
    reason[any_biv & tall_baby] = "age_vs_size"
    reason[any_biv & suspect] = "age_suspect"

    action = pd.Series("keep", index=out.index, dtype="object")
    action[any_biv] = "review"
    action[reason.isin(["age_suspect", "age_vs_size", "contradictory"])] = "remove"

    out["biv_any"] = any_biv
    out["biv_reason"] = reason.replace({"other_biv": "no_rule"})
    out["biv_action"] = action
    out["biv_removed"] = action.eq("remove") | (action.eq("review") if remove_review else False)
    # how many of the three rules each row matches (rows can match several; reason keeps the first)
    out["biv_rules_matched"] = (any_biv & suspect).astype(int) + (any_biv & tall_baby).astype(int) \
        + (any_biv & contradictory).astype(int)
    return out


def biv_summary(df: pd.DataFrame) -> None:
    """Counts to paste into the report."""
    sc = df[df["growth_status"] == "scored"]
    print("Scored rows:", len(sc))
    print("BIV-flagged:", int(sc["biv_any"].sum()))
    print("\nFirst-matching rule (each flagged row counted once):")
    print(sc.loc[sc["biv_any"], "biv_reason"].value_counts().to_string())
    print("\nAction:")
    print(sc["biv_action"].value_counts().to_string())
    print("\nRemoved by reference:")
    print(sc[sc["biv_removed"]].groupby("growth_ref").size().to_string())
    print("\nRows matching more than one rule:", int((sc["biv_rules_matched"] > 1).sum()))
    r = sc[sc["biv_action"] == "review"]
    if "normal_x_anormal" in r:
        print("\nTo review, by label:", r["normal_x_anormal"].value_counts().to_dict())
    print("\nRows after removal:", int((~df["biv_removed"]).sum()), "of", len(df))


REVIEW_COLS = [
    # who / what age
    "sexo", "age_years", "idade_origem", "growth_ref",
    # raw measures
    "altura", "peso", "imc",
    # z-scores (how far from a typical child of that age and sex)
    "haz", "waz", "bmiz", "whz",
    # percentiles
    "hapct", "wapct", "bmipct", "whpct",
    # clinical context: is a real extreme plausible for this child?
    "normal_x_anormal", "motivo_1", "motivo_2", "hda_1", "sopro", "fc",
    "pa_sistolica", "pa_diastolica",
]


def _why_flagged(df: pd.DataFrame) -> pd.Series:
    """Plain-text reason, e.g. 'height HIGH (z=5.2); weight HIGH (z=9.8)'."""
    parts = pd.DataFrame(index=df.index)
    for flag, z, name in (("biv_ht", "haz", "height"), ("biv_wt", "waz", "weight"),
                          ("biv_bmi", "bmiz", "BMI"), ("biv_wh", "whz", "weight-for-height")):
        f, v = df[flag], df[z]
        txt = np.where(f > 0, name + " HIGH (z=" + v.round(1).astype(str) + ")",
                       np.where(f < 0, name + " LOW (z=" + v.round(1).astype(str) + ")", ""))
        parts[name] = txt
    return parts.apply(lambda r: "; ".join(x for x in r if x), axis=1)


def export_review(df: pd.DataFrame, path: str, cols: list[str] | None = None) -> pd.DataFrame:
    """Write the rows with biv_action == 'review' to a CSV so you can judge them by hand.

    Columns: row_id | decision (you fill: remove / keep) | note (your reason, optional) |
    why_flagged | measures, z-scores, percentiles and clinical context.
    Only `row_id` and `decision` are read back by apply_manual_decisions()."""
    cols = [c for c in (cols or REVIEW_COLS) if c in df]
    sub = df.loc[df["biv_action"] == "review"]
    r = sub[cols].round(2)
    r.insert(0, "why_flagged", _why_flagged(sub))
    r.insert(0, "note", "")
    r.insert(0, "decision", "")
    r.insert(0, "row_id", r.index)
    r = r.sort_values(["growth_ref", "age_years"] if "age_years" in r else ["growth_ref"])
    r.to_csv(path, index=False)
    print(f"{len(r)} rows written to {path}")
    return r


def apply_manual_decisions(df: pd.DataFrame, path: str) -> pd.DataFrame:
    """Read the reviewed CSV back: rows with decision == 'remove' get biv_removed = True,
    reason 'manual_review'.  Rows left blank or 'keep' stay in the data."""
    d = pd.read_csv(path)
    ids = d.loc[d["decision"].astype(str).str.lower().eq("remove"), "row_id"]
    out = df.copy()
    out.loc[ids, "biv_removed"] = True
    out.loc[ids, "biv_reason"] = "manual_review"
    return out