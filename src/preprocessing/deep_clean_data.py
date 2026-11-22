import pandas as pd
from growth_scores import load_refs, add_growth_scores
from biv_removal import add_biv_decision, biv_summary, export_review

df = pd.read_csv("data/UCMF_labelled.csv")
refs = load_refs("src/preprocessing/WHOref_d.sas7bdat",
                 "src/preprocessing/CDCref_d.sas7bdat")

out = add_growth_scores(df, refs, sex_col="sexo", birth_col="data_nascimento",
                        visit_col="atendimento", height_col="altura",
                        weight_col="peso", date_format="%Y-%m-%d")
out = add_biv_decision(out)
biv_summary(out)
export_review(out, "data/biv_to_review.csv")   # judge these 188 rows by hand

# After filling the `decision` column in that CSV, add:
# from biv_removal import apply_manual_decisions
# out = apply_manual_decisions(out, "data/biv_to_review.csv")

out.to_csv("data/UCMF_annotated.csv", index=False)

bookkeeping = ["biv_ht", "biv_wt", "biv_bmi", "biv_wh", "biv_low", "biv_high",
               "biv_any", "biv_reason", "biv_action", "biv_removed", "biv_rules_matched"]
clean = out[~out["biv_removed"]].drop(columns=bookkeeping)
clean.to_csv("data/UCMF_clean.csv", index=False)