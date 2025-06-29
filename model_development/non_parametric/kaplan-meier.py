import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from pathlib import Path

# --------------- CONFIG ----------------
DATA_PATH = Path("data/train_trials.csv")
SUMMARY_PATH = Path("data/km_summary.csv")
FIGURE_PATH = Path("figure/km_survival_by_phase.png")

PHASE_COLS = [
    "phase_early_phase1", "phase_phase1", "phase_phase2",
    "phase_phase3", "phase_phase4"
]

# ------------ GET PHASE LABEL ------------
def get_phase(row):
    for col in reversed(PHASE_COLS):  # Phase 4 > Phase 3 > ...
        if row.get(col, 0) == 1:
            return col.replace("phase_", "").upper()
    return "UNKNOWN"

# ------------ MAIN FUNCTION ------------
def run_kaplan_meier():
    df = pd.read_csv(DATA_PATH)
    df = df[df["trial_duration_days"].notnull()]
    T = df["trial_duration_days"].astype(int).values
    E = df["status"].astype(int).values  # 1 = failed, 0 = completed (already encoded correctly)

    sponsor_label = df["sponsor_class_category_industry"].apply(lambda x: "Industry" if x == 1 else "Non-Industry")
    phase_label = df[PHASE_COLS].apply(get_phase, axis=1)

    unique_phases = sorted(phase_label.unique())
    rows, cols = 2, 3
    fig, axs = plt.subplots(rows, cols, figsize=(6 * cols, 4.5 * rows))
    axs = axs.flatten()

    summary = []
    for i, phase in enumerate(unique_phases):
        ax = axs[i]
        phase_mask = (phase_label == phase)

        for sponsor in ["Industry", "Non-Industry"]:
            sponsor_mask = (sponsor_label == sponsor)
            mask = phase_mask & sponsor_mask

            if mask.sum() == 0:
                continue

            kmf = KaplanMeierFitter()
            kmf.fit(T[mask], E[mask], label=sponsor)
            kmf.plot_survival_function(ax=ax)

            n = mask.sum()
            events = E[mask].sum()
            censoring = 1 - (events / n)
            median = kmf.median_survival_time_
            try:
                q75 = kmf.percentile(75)
            except:
                q75 = np.nan

            summary.append({
                "Phase": phase,
                "Sponsor Type": sponsor,
                "N": int(n),
                "Events": int(events),
                "Censoring Rate": f"{censoring:.2%}",
                "Median Survival": f"{median:.1f}" if np.isfinite(median) else "Not estimable",
                "75th Percentile": f"{q75:.1f}" if np.isfinite(q75) else "Not estimable"
            })

        # Log-rank test
        m1 = phase_mask & (sponsor_label == "Industry")
        m2 = phase_mask & (sponsor_label == "Non-Industry")
        if m1.sum() > 0 and m2.sum() > 0:
            p = logrank_test(T[m1], T[m2], E[m1], E[m2]).p_value
            ax.text(0.6, 0.2, f"Log-rank p = {p:.4f}", transform=ax.transAxes)

        ax.set_title(f"Phase: {phase}")
        ax.set_xlabel("Trial Duration (days)")
        ax.set_ylabel("Survival Probability")
        ax.grid(True)

    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle("Kaplan-Meier Survival by Phase and Sponsor Type", fontsize=16)

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURE_PATH)
    plt.close()

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(SUMMARY_PATH, index=False)

    print(f"Figure saved to: {FIGURE_PATH}")
    print(f"Summary saved to: {SUMMARY_PATH}")

# ------------ RUN ------------
if __name__ == "__main__":
    run_kaplan_meier()
