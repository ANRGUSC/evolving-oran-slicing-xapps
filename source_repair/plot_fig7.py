"""Render the submitted paper's single-panel Fig. 7 from recomputed trace scores."""
import argparse
import json
import math
from pathlib import Path


def load_pairs(path):
    """Validate the complete paired 16-scenario experiment before drawing."""
    pairs = json.loads(Path(path).read_text())["pairs"]
    if len(pairs) != 16 or [row["dev_seed"] for row in pairs] != list(range(16)):
        raise ValueError("Figure 7 requires development scenarios 0 through 15 in order")
    for row in pairs:
        for name in ("original", "edited"):
            if type(row[name]) not in (int, float) or not math.isfinite(row[name]):
                raise ValueError(f"Invalid {name} score for scenario {row['dev_seed']}")
    return pairs


def build_figure(pairs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Match the submitted figure's content: orange circles, green diamonds,
    # paired vertical guides, and one point per development scenario.
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 8,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False}):
        fig, ax = plt.subplots(figsize=(3.5, 2.3))
        x = [row["dev_seed"] for row in pairs]
        before = [row["original"] for row in pairs]
        after = [row["edited"] for row in pairs]
        ax.vlines(x, before, after, color="#cbd2d8", linewidth=0.7, zorder=1)
        ax.scatter(x, before, s=16, color="#d66b24", marker="o",
                   label="Original", zorder=3)
        ax.scatter(x, after, s=18, color="#4a9e84", marker="D",
                   label="Edited", zorder=3)
        ax.set(xlabel="Channel scenario (seed 0–15)",
               ylabel="Episode-average reward", xlim=(-0.6, 15.6))
        ax.set_ylim(min(0.26, min(before + after) - 0.03),
                    max(0.86, max(before + after) + 0.03))
        ax.set_xticks(x)
        ax.tick_params(axis="x", labelsize=6.5)
        ax.grid(axis="y", color="#e0e5e9", linewidth=0.6)
        ax.legend(loc="center right", frameon=False)
        fig.tight_layout(pad=0.5)
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    fig = build_figure(load_pairs(args.data))
    args.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out / "fig7.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(args.out / "fig7.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
    import matplotlib.pyplot as plt
    plt.close(fig)
    print("Created fig7.pdf and fig7.png (submitted single-panel layout)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
