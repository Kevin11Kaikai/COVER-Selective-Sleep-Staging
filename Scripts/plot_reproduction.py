"""Plot the recomputed operating points, without reading printed paper values."""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def draw(out):
    with (out / "main/own_rule_summary.csv").open() as stream:
        main = list(csv.DictReader(stream))
    with (out / "development/summary.csv").open() as stream:
        nested = [r for r in csv.DictReader(stream) if r["branch"] == "nested"
                  and r["evaluation"] not in ("oracle", "quota_0.4", "quota_0.5", "quota_0.6")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    colors = {"Linear-BCE": "#b5473e", "FixedScale": "#1869aa", "SELE": "#8d489b"}
    for row in main:
        if row["method"] == "Learned-Selected":
            continue
        label = row["method"]
        x, y = 100 * float(row["macro_coverage"]), 100 * float(row["subject_equal_ratio_risk"])
        axes[0].scatter(x, y, color=colors.get(label, "#69777b"), s=40)
        axes[0].annotate(label, (x, y), xytext=(3, 6), textcoords="offset points", fontsize=8)
    for row in nested:
        label = row["cell"].removeprefix("Nested-")
        if label not in ("BCE", "SELE", "FixedScale-1"):
            continue
        label = {"BCE": "Linear-BCE", "FixedScale-1": "FixedScale"}.get(label, label)
        x, y = 100 * float(row["coverage"]), 100 * float(row["R_SE"])
        axes[1].scatter(x, y, color=colors[label], s=50)
        axes[1].annotate(label, (x, y), xytext=(3, 6), textcoords="offset points", fontsize=9)
    for ax, title in zip(axes, ("Main evaluation", "Nested development (same cohort)")):
        ax.set(title=title, xlabel="Mean subject coverage (%)", ylabel="Subject-equal accepted risk (%)")
        ax.axhline(5, color="gray", ls="--", lw=0.8)
        ax.grid(alpha=0.2)
        ax.margins(x=0.25, y=0.2)
    folder = out / "figures"
    folder.mkdir(exist_ok=True)
    fig.savefig(folder / "operating_points.png", dpi=160)
    fig.savefig(folder / "operating_points.pdf")
    plt.close(fig)
