"""Reaggregate saved subject records; this does not refit nested models."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from new_objective.bceanchor_upgrade_report import aggregate_people, summarize, paired_tables


def run(root: Path, out: Path):
    frozen = root / "artifacts/frozen/New_Objective_Idea/outputs/bceanchor_upgrade/tables"
    per = pd.read_csv(frozen / "per_subject_records.csv",
                      dtype={"context": str, "subject": str, "risk_fold": str})
    keys = ["branch", "cell", "evaluation", "context", "subject"]
    if per.duplicated(keys).any():
        raise ValueError("Duplicate subject/context records")
    if not ((per.errors >= 0) & (per.errors <= per.accepted) &
            (per.accepted <= per.n) & (per.n > 0)).all():
        raise ValueError("Invalid saved counts")
    np.testing.assert_allclose(per.coverage, per.accepted / per.n, atol=1e-15, rtol=0)
    np.testing.assert_allclose(per.error_mass, per.errors / per.n, atol=1e-15, rtol=0)
    # Memberships are checked against the frozen nested BCE reference, not assumed
    # to occur 15 times per person; context assignment is not globally balanced.
    baseline = per[(per.branch == "nested") & (per.cell == "Nested-BCE")]
    baseline = baseline[baseline.evaluation == sorted(baseline.evaluation.unique())[0]]
    if len(baseline) != 300 or baseline.context.nunique() != 20:
        raise ValueError("Expected 20 FIT contexts with 15 subjects each")
    if not (baseline.groupby("context").subject.nunique() == 15).all():
        raise ValueError("Incomplete FIT context")
    expected = {subject: sorted(g.context.tolist()) for subject, g in baseline.groupby("subject")}
    if len(expected) != 20 or any(s in cs for s, cs in expected.items()):
        raise ValueError("Invalid development/OUTER role memberships")
    people = aggregate_people(per)
    summary = summarize(people, expected)
    cfg = json.loads((root / "New_Objective_Idea/bceanchor_upgrade_protocol.json").read_text())
    paired, deletion = paired_tables(people, cfg, expected)
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in (("subject_means", people), ("summary", summary),
                        ("paired_differences", paired), ("delete_one_subject", deletion)):
        frame.to_csv(out / (name + ".csv"), index=False, float_format="%.17g")
    return {"input_records": len(per), "summary_rows": len(summary),
            "subjects": len(expected), "contexts": 20, "refitting": False}
