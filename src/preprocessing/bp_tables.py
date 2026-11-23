"""
NHBPEP 2004 (Fourth Report) BP tables for children, built from the project's blood_pressure_children.pdf.
Source: 'The Fourth Report on the Diagnosis, Evaluation, and Treatment of High Blood Pressure in Children
and Adolescents', Pediatrics 2004;114:555-576, Tables 3 and 4 (ages 1-17).

load_bp_tables() -> DataFrame, one row per (sex, age, bp_pct):
    sex (1 boy / 2 girl), age (1-17), bp_pct (50/90/95/99),
    sbp_h5..sbp_h95 and dbp_h5..dbp_h95   (height percentile columns 5,10,25,50,75,90,95)
"""
import re
from pathlib import Path
import pandas as pd

HEIGHT_PCT = [5, 10, 25, 50, 75, 90, 95]
_RAW = Path(__file__).with_name("bp_tables_raw.txt")


def load_bp_tables(path=_RAW) -> pd.DataFrame:
    rows, sex, age = [], None, None
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line.startswith("#"):
            sex, age = (1 if "BOYS" in line else 2), None
            continue
        m = re.match(r"^(?:(\d+)\s+)?(50|90|95|99)th((?:\s+\d+){14})$", line)
        if not m:
            continue
        if m.group(1):
            age = int(m.group(1))
        v = [int(x) for x in m.group(3).split()]
        rows.append({"sex": sex, "age": age, "bp_pct": int(m.group(2)),
                     **{f"sbp_h{h}": v[i] for i, h in enumerate(HEIGHT_PCT)},
                     **{f"dbp_h{h}": v[7 + i] for i, h in enumerate(HEIGHT_PCT)}})
    t = pd.DataFrame(rows)
    assert len(t) == 2 * 17 * 4, len(t)
    return t
