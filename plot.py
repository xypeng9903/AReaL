import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame


# -----------------------------
# User settings
# -----------------------------
SAVE_PATH = "plot_train-dapo17k.png"
ROOT = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/pengxinyu05/training-runs"
EXP_NAMES = [
    "areal-veomini/0307-areal_fsdp-dapo17k_dppo_binary_kl-qwen3-8b",
    "areal-veomini/0307-areal_veomini-dapo17k_dppo_binary_kl-qwen3-8b--veomini_engine_debug",
    "areal-veomini/0312-areal_fsdp-dapo17k_dppo_binary_kl-qwen3-8b--fp32-master-weight",
    "areal-veomini/0313-areal_fsdp-dapo17k_dppo_binary_kl-qwen3-8b--fp32-master-weight--lr-3e-6",
    "areal-veomini/0313-areal_veomini-dapo17k_dppo_binary_kl-qwen3-8b--fp32-master-weight--lr-3e-6",
]

METRIC_KEYS: list[dict[str, Any]] = [
    {"name": "eval-rollout/reward", "ylim": None, "log_y": False, "log_x": False},
    {"name": "rollout/reward", "ylim": None},
    {"name": "ppo_actor/seq_len/avg", "ylim": None},
    {"name": "ppo_actor/update/grad_norm", "ylim": None},
    {"name": "ppo_actor/update/importance_weight/avg", "ylim": None},
    {"name": "ppo_actor/update/importance_weight/min", "ylim": None},
    {"name": "ppo_actor/update/importance_weight/max", "ylim": None},
    {"name": "ppo_actor/update/version_stats/sample_staleness_theta_avg", "ylim": None},
    # {"name": "ppo_actor/update/behave_imp_weight/avg", "ylim": None},
    # {"name": "ppo_actor/update/behave_imp_weight/max", "ylim": None},
    # {"name": "ppo_actor/update/behave_imp_weight/min", "ylim": None},
]

# TensorBoard-like smoothing (0 means no smoothing, closer to 1 means stronger smoothing)
SMOOTHING = 0.5
# Show raw curves as faint background for context
SHOW_RAW = True
# Layout
MAX_COLS = 3
# Optional point cap for faster plotting on very long runs
MAX_POINTS_PER_SERIES = 6000


def _find_metrics_file(run_path: Path) -> Path | None:
    metric_files = list(run_path.glob("**/metrics.jsonl"))
    if not metric_files:
        return None
    # Prefer latest modified metrics file when there are multiple candidates
    return max(metric_files, key=lambda p: p.stat().st_mtime)


def _load_metrics_df(metrics_path: Path) -> DataFrame:
    rows: list[dict[str, Any]] = []
    with metrics_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                print(f"[{metrics_path}, line {line_no}] {e}")

    if not rows:
        return DataFrame()

    df = DataFrame(rows)
    if "global_step" not in df.columns:
        print(f"[WARN] global_step missing in {metrics_path}")
        return DataFrame()

    df = df.drop_duplicates(subset=["global_step"], keep="first")
    df = df.sort_values("global_step")
    return df


def _ema(values: np.ndarray, smoothing: float) -> np.ndarray:
    if len(values) == 0 or smoothing <= 0:
        return values
    smoothing = min(max(smoothing, 0.0), 0.999)

    smoothed = np.empty_like(values, dtype=np.float64)
    smoothed[0] = float(values[0])
    for i in range(1, len(values)):
        smoothed[i] = smoothed[i - 1] * smoothing + values[i] * (1.0 - smoothing)
    return smoothed


def _downsample_xy(x: np.ndarray, y: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if len(x) <= max_points:
        return x, y
    idx = np.linspace(0, len(x) - 1, max_points, dtype=int)
    return x[idx], y[idx]


def _auto_layout(n_panels: int, max_cols: int = 3) -> tuple[int, int]:
    cols = min(max_cols, int(np.ceil(np.sqrt(n_panels))))
    rows = int(np.ceil(n_panels / cols))
    return rows, cols


def main() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    root = Path(ROOT)
    n_metrics = len(METRIC_KEYS)
    n_rows, n_cols = _auto_layout(n_metrics, max_cols=MAX_COLS)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6.2 * n_cols, 3.8 * n_rows),
        constrained_layout=False,
    )
    axes_arr = np.atleast_1d(axes).flatten()

    exp_dfs: dict[str, DataFrame] = {}
    for exp_name in EXP_NAMES:
        run_path = root / exp_name
        metrics_path = _find_metrics_file(run_path)
        if metrics_path is None:
            print(f"[WARN] metrics.jsonl not found in {run_path}")
            continue
        print(f"Plotting {metrics_path}")
        df = _load_metrics_df(metrics_path)
        if df.empty:
            print(f"[WARN] empty/invalid metrics for {exp_name}")
            continue
        exp_dfs[exp_name] = df

    if not exp_dfs:
        print("No valid experiment data found.")
        return

    palette = plt.get_cmap("tab10")

    for i, metric in enumerate(METRIC_KEYS):
        ax = axes_arr[i]
        metric_name = metric["name"]

        for exp_idx, (exp_name, df) in enumerate(exp_dfs.items()):
            if metric_name not in df.columns:
                continue

            subset = df[["global_step", metric_name]].dropna().sort_values("global_step")
            if subset.empty:
                continue

            x = np.asarray(subset["global_step"], dtype=np.float64)
            y = np.asarray(subset[metric_name], dtype=np.float64)
            x, y = _downsample_xy(x, y, MAX_POINTS_PER_SERIES)

            color = palette(exp_idx % 10)

            if SHOW_RAW:
                ax.plot(x, y, color=color, alpha=0.18, linewidth=1.0)

            y_smooth = _ema(y, SMOOTHING)
            ax.plot(
                x,
                y_smooth,
                label=exp_name,
                color=color,
                linewidth=1.8,
            )

        ax.set_title(metric_name, fontsize=10)
        ax.set_xlabel("global_step")
        ax.set_ylabel("value")
        ax.grid(True, which="major", linestyle="--", alpha=0.32)
        ax.grid(True, which="minor", linestyle=":", alpha=0.18)

        if metric.get("ylim") is not None:
            ax.set_ylim(metric["ylim"])
        if metric.get("log_y", False):
            ax.set_yscale("log")
        if metric.get("log_x", False):
            ax.set_xscale("log")

    # Hide unused axes in grid
    for j in range(n_metrics, len(axes_arr)):
        axes_arr[j].axis("off")

    # One global legend: cleaner than repeating legend in every subplot
    handles, labels = [], []
    for ax in axes_arr:
        h, l = ax.get_legend_handles_labels()
        for idx in range(len(h)):
            if l[idx] not in labels:
                variables_tuple = (h[idx], l[idx])
                handles.append(variables_tuple[0])
                labels.append(variables_tuple[1])

    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=min(3, len(labels)),
            fontsize=9,
            frameon=True,
        )

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(SAVE_PATH, dpi=220, bbox_inches="tight")
    print(f"Saved figure to: {SAVE_PATH}")


if __name__ == "__main__":
    main()
