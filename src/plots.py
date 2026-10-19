"""Figures for the poster / README (matplotlib only)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from preprocess import VITALS, UNITS

COLORS = {"artifact": "#e08a00", "real_event": "#c0392b", "review": "#7f8c8d"}


def plot_case(df, episodes, caseid, path, hours=None):
    """Five vitals of one patient with detected episodes shaded (orange = artifact, red = real event)."""
    d = df[df.caseid == caseid]
    e = episodes[episodes.caseid == caseid]
    fig, axes = plt.subplots(5, 1, figsize=(13, 9), sharex=True)
    for ax, v in zip(axes, VITALS):
        ax.plot(d.time_s / 60, d[v], lw=0.7, color="#2c3e50")
        for _, r in e.iterrows():
            if r["label"] == "review":
                continue
            ax.axvspan(r.start_s / 60 - 0.3, r.end_s / 60 + 0.3, color=COLORS[r["label"]],
                       alpha=0.9 if r["vital"] == v else 0.25, lw=0)
        ax.set_ylabel(f"{v}\n({UNITS[v]})")
    axes[0].set_title(f"Case {caseid}: detected episodes (orange = artifact, red = real event)")
    axes[-1].set_xlabel("Minutes since recording start")
    plt.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_class_mix(episodes, path):
    tab = episodes.groupby(["vital", "label"]).size().unstack(fill_value=0).reindex(VITALS).fillna(0)
    fig, ax = plt.subplots(figsize=(7, 4)); bottom = np.zeros(len(tab))
    for lab in ["artifact", "real_event", "review"]:
        if lab in tab:
            ax.bar(tab.index, tab[lab], bottom=bottom, label=lab.replace("_", " "), color=COLORS[lab]); bottom += tab[lab].to_numpy()
    ax.set_ylabel("Episodes"); ax.set_title("Detected episodes per vital"); ax.legend()
    plt.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_confusion(conf, path):
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    im = ax.imshow(conf.to_numpy(), cmap="Blues")
    ax.set_xticks(range(conf.shape[1]), [c.replace("pred_", "").replace("_", " ") for c in conf.columns])
    ax.set_yticks(range(conf.shape[0]), [i.replace("true_", "").replace("_", " ") for i in conf.index])
    for i in range(conf.shape[0]):
        for j in range(conf.shape[1]):
            ax.text(j, i, int(conf.iat[i, j]), ha="center", va="center", fontsize=12)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Injected truth"); ax.set_title("Artifact vs real-event separation (test patients)")
    plt.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_sensitivity(sens, path):
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(sens.K, sens.detection_recall, marker="o", label="injected episodes detected")
    ax.plot(sens.K, sens.correct_class_rate, marker="s", label="detected AND classified correctly")
    ax.set_xlabel("Threshold K (x patient-specific scale)"); ax.set_ylabel("Fraction"); ax.set_ylim(0, 1.02)
    ax2 = ax.twinx(); ax2.plot(sens.K, sens.background_episodes_per_hour, color="grey", ls="--", marker="^")
    ax2.set_ylabel("Unmatched episodes per hour (dashed)")
    ax.legend(loc="lower left"); ax.set_title("Effect of the threshold K")
    plt.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
