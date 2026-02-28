import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import os
from typing import Optional, Dict

"""
plot_model_validations
------------------------
Visualise cross-validated metrics for multiple regression models
across multiple prediction targets.

Encoding (3 independent performance dimensions):
  X position  – MAPE_%  (or any configured x_metric)
  Y position  – Pearson r  (auto-selected when target CV < cv_threshold)
  Bubble area – MaxAPE_%  (worst-case / tail error)
  Colour      – Model identity  (all circles — shape carries no info)
  Label       – Model name  directly on point (no legend needed)

Overlap handling:
  Labels are placed with smart directional offsets. When two points are
  within `overlap_threshold` in normalised plot space, their label offsets
  are pushed apart radially so neither obscures the other.

Design:
  - No shape variety (all 'o') — colour alone identifies each model
  - No legend for models — direct annotation is cleaner for essays
  - Size legend in bottom-right corner shows MaxAPE reference bubbles
  - Drop-lines to both axes (colour-matched, darkened, dashed)
  - No axhline/axvline — pure data-space lines (avoids mpld3 warning)
  - Uses plt.colormaps (no deprecated get_cmap)
  - layout='constrained' (no tight_layout warning)
"""

# ── Bubble sizing constants ──────────────────────────────────────────────────
_BUBBLE_MIN = 120  # minimum marker area (pt²) — always visible
_BUBBLE_MAX = 900  # maximum marker area (pt²)

# ── Label overlap detection ──────────────────────────────────────────────────
_OVERLAP_THRESH = 0.18  # normalised distance below which labels are pushed apart

# ── Base label offset in data-fraction units (before overlap adjustment) ────
_LABEL_OFF_X = 0.04  # prefer a slightly larger horizontal offset (top-right)
_LABEL_OFF_Y = 0.045  # prefer a slightly larger vertical offset (top-right)


def _deepen_color(c, factor: float = 0.72):
    """Return a darker version of a matplotlib colour spec."""
    r, g, b, a = mcolors.to_rgba(c)
    return (r * factor, g * factor, b * factor, a)


def _bubble_sizes(max_ape_vals: list,
                  s_min: float = _BUBBLE_MIN,
                  s_max: float = _BUBBLE_MAX) -> np.ndarray:
    """
    Map MaxAPE values to marker areas.
    Area scales linearly with MaxAPE so larger bubbles = worse tail error.
    Falls back to uniform size if MaxAPE is unavailable or all equal.
    """
    vals = np.array(max_ape_vals, dtype=float)
    vmin, vmax = vals.min(), vals.max()
    if vmax - vmin < 1e-9:
        return np.full(len(vals), (s_min + s_max) / 2)
    return s_min + (vals - vmin) / (vmax - vmin) * (s_max - s_min)


def _smart_offsets(model_data: list, x_key: str, y_key: str,
                   x_range: float, y_range: float,
                   base_ox: float, base_oy: float,
                   thresh: float = _OVERLAP_THRESH) -> dict:
    """
    Compute per-model (dx, dy) label offsets in data units.

    Strategy:
      1. Each point starts with a default offset (away from plot centre).
      2. For every pair closer than `thresh` in normalised space, the offset
         vectors are reflected so they point in opposite directions — one up,
         one down — creating clean vertical separation of the labels.
    """
    n = len(model_data)
    names = [d['model'] for d in model_data]
    xs = np.array([d[x_key] for d in model_data])
    ys = np.array([d[y_key] for d in model_data])

    # Default offset: right-and-up
    offsets = {m: np.array([base_ox * x_range,
                            base_oy * y_range]) for m in names}

    # Detect close pairs and push them apart
    for i in range(n):
        for j in range(i + 1, n):
            dx_norm = (xs[i] - xs[j]) / (x_range or 1)
            dy_norm = (ys[i] - ys[j]) / (y_range or 1)
            dist = np.sqrt(dx_norm ** 2 + dy_norm ** 2)
            if dist < thresh:
                # Push i up, j down (vertical split is clearest)
                offsets[names[i]] = np.array([base_ox * x_range,
                                              base_oy * y_range * 2.2])
                offsets[names[j]] = np.array([base_ox * x_range,
                                              -base_oy * y_range * 2.2])

    return offsets


def _pareto_front(model_data: list, x_key: str, y_key: str) -> list:
    """Return indices of Pareto-non-dominated points (min x, max y)."""
    n = len(model_data)
    dominated = set()
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if (model_data[j][x_key] <= model_data[i][x_key] and
                    model_data[j][y_key] >= model_data[i][y_key] and
                    (model_data[j][x_key] < model_data[i][x_key] or
                     model_data[j][y_key] > model_data[i][y_key])):
                dominated.add(i)
                break
    return [i for i in range(n) if i not in dominated]


def plot_model_validations(
        results_dict: dict,  # CHANGED: now results_dict[target][model]
        output_dir: str = "./validation_plots",
        figsize: tuple = (8, 6),
        dpi: int = 150,
        target_cvs: Optional[dict] = None,
        cv_threshold: float = 0.2,
        x_metric: str = "MAPE_%",
        y_metric: Optional[str] = None,
        size_metric: str = "MaxAPE_%",
        overlap_threshold: float = _OVERLAP_THRESH,
) -> dict:
    """
    Parameters
    ----------
    results_dict : dict
        results_dict[target_name][model_name] = {metric_key: value, ...}
        Common keys: "MAPE_%", "CVRMSE_%", "MaxAPE_%", "Pearson_r", "R2"

    output_dir : str
        Directory for saved PNG files.

    figsize, dpi : figure size and resolution.

    target_cvs : dict, optional
        {target_name: CV_float}  e.g. {"Max Mises(Mpa)": 0.062}
        Drives automatic Y-axis selection.

    cv_threshold : float  (default 0.2)
        When target CV < this, Y-axis switches to Pearson_r.

    x_metric : str  (default "MAPE_%")
        Metric key to use on the X axis.

    y_metric : str or None
        Force a Y-axis metric for all targets. None = auto via cv_threshold.

    size_metric : str  (default "MaxAPE_%")
        Metric key whose value maps to bubble area (3rd dimension).
        If absent in data, all bubbles are drawn at uniform size.

    overlap_threshold : float  (default 0.18)
        Normalised-space distance below which label offsets are adjusted.

    Returns
    -------
    summary : dict
        {target: {"pareto_models", "top3_models", "x_axis", "y_axis"}}
    """
    os.makedirs(output_dir, exist_ok=True)

    # CHANGED: Get targets directly from keys, models from first target's keys
    targets = list(results_dict.keys())
    models = list(next(iter(results_dict.values())).keys())

    target_cvs = target_cvs or {}

    # ── palette: all circles, distinct colours ───────────────────────────────
    tab10 = plt.colormaps['tab10']
    colors = {m: tab10(i % 10) for i, m in enumerate(models)}

    # CHANGED: Get sample from first target, first model
    sample = next(iter(next(iter(results_dict.values())).values()))
    has_pearson = "Pearson_r" in sample
    has_size = size_metric in sample

    summary = {}

    for target in targets:

        # ── resolve axis keys ────────────────────────────────────────────────
        x_key = x_metric
        if y_metric is not None:
            y_key = y_metric
        else:
            cv_val = target_cvs.get(target, 1.0)
            y_key = ("Pearson_r"
                     if has_pearson and cv_val < cv_threshold
                     else "R2")

        x_label = x_key.replace("_%", " (%)").replace("_", " ")
        y_label = y_key.replace("_", " ")

        # ── collect data ─────────────────────────────────────────────────────
        # CHANGED: iterate over models for this target
        model_data = []
        for m in models:
            e = results_dict[target][m]  # CHANGED: [target][m] instead of [m][target]
            model_data.append({
                'model': m,
                x_key: e[x_key],
                y_key: e[y_key],
                size_metric: e.get(size_metric, None),
                'combined_score': e[x_key] - e[y_key] * 10,
            })

        # ── bubble sizes (MaxAPE → area) ─────────────────────────────────────
        raw_sizes = [d[size_metric] for d in model_data]
        if has_size and all(v is not None for v in raw_sizes):
            sizes = _bubble_sizes(raw_sizes)
        else:
            sizes = np.full(len(model_data), (_BUBBLE_MIN + _BUBBLE_MAX) / 2)

        # ── Pareto ───────────────────────────────────────────────────────────
        pareto_indices = _pareto_front(model_data, x_key, y_key)
        pareto_models = sorted(
            [model_data[i] for i in pareto_indices],
            key=lambda d: d[x_key]
        )
        pareto_names = {m['model'] for m in pareto_models}
        top3 = sorted(pareto_models, key=lambda d: d['combined_score'])[:3]

        # ── axis limits (computed before drawing) ────────────────────────────
        all_x = [d[x_key] for d in model_data]
        all_y = [d[y_key] for d in model_data]
        x_range = max(all_x) - min(all_x) or 1.0
        y_range = max(all_y) - min(all_y) or 1.0

        # Extra padding: right side needs more room for labels
        x_lo = min(all_x) - x_range * 0.18
        x_hi = max(all_x) + x_range * 0.38
        y_lo = min(all_y) - y_range * 0.28
        y_hi = max(all_y) + y_range * 0.28

        # ── smart label offsets ───────────────────────────────────────────────
        offsets = _smart_offsets(
            model_data, x_key, y_key, x_range, y_range,
            _LABEL_OFF_X, _LABEL_OFF_Y, thresh=overlap_threshold
        )

        # ── figure ───────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=figsize, layout='constrained')
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)

        # ── drop lines to axes (pure data-space, colour-matched) ─────────────
        for d in model_data:
            col = colors[d['model']]
            # make the drop/dashed cast lines slightly deeper for legibility
            drop_col = _deepen_color(col, 0.60)
            px, py = d[x_key], d[y_key]
            ax.plot([px, px], [y_lo, py],
                    color=drop_col, linestyle='--',
                    linewidth=0.6, alpha=0.50, zorder=2)
            ax.plot([x_lo, px], [py, py],
                    color=drop_col, linestyle='--',
                    linewidth=0.6, alpha=0.50, zorder=2)

        # ── bubbles: all circles ──────────────────────────────────────────────
        for d, sz in zip(model_data, sizes):
            col = colors[d['model']]
            ax.scatter(
                d[x_key], d[y_key],
                s=sz,
                color=col,
                marker='o',
                edgecolors=_deepen_color(col, 0.65),
                linewidths=0.8,
                alpha=0.88,
                zorder=5,
            )

        # ── direct model name labels (only for top candidates) ──────────────
        # Show only the model name next to each top candidate so the plot
        # remains uncluttered. Descriptive metric text for these models is
        # listed in a single horizontal line at the top-left of the axes.
        for m in top3:
            col = colors[m['model']]
            ox, oy = offsets[m['model']]
            px, py = m[x_key], m[y_key]
            tx, ty = px + ox * 0.7, py + oy * 0.7

            ax.text(
                tx, ty,
                m['model'],
                fontsize=8,
                color=_deepen_color(col, 0.28),
                fontweight='semibold',
                va='bottom', ha='left',
                zorder=8,
            )

        # ── threshold reference lines (pure data-space — no blended transform) ─
        if y_key == "Pearson_r":
            for rv, lc in [(0.90, '#2ca02c'), (0.95, '#1a7a1a')]:
                if y_lo < rv < y_hi:
                    ax.plot([x_lo, x_hi], [rv, rv],
                            linestyle=':', linewidth=0.85,
                            color=lc, alpha=0.50, zorder=1)
                    ax.text(x_lo + (x_hi - x_lo) * 0.01, rv,
                            f'r = {rv}', va='bottom', ha='left',
                            fontsize=7, color=lc, alpha=0.80)
        elif y_key == "R2":
            if y_lo < 0.0 < y_hi:
                ax.plot([x_lo, x_hi], [0.0, 0.0],
                        linestyle=':', linewidth=0.85,
                        color='#d62728', alpha=0.50, zorder=1)
                ax.text(x_lo + (x_hi - x_lo) * 0.01, 0.0,
                        'R² = 0', va='bottom', ha='left',
                        fontsize=7, color='#d62728', alpha=0.80)

        # ── cosmetics ────────────────────────────────────────────────────────
        ax.set_xlabel(x_label, fontsize=11, labelpad=6)
        ax.set_ylabel(y_label, fontsize=11, labelpad=6)
        ax.set_title(f'{target}  —  {x_label} vs {y_label}',
                     fontsize=12, pad=10)
        ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.30, zorder=0)
        ax.spines[['top', 'right']].set_visible(False)

        # ── bubble size legend (MaxAPE reference) ────────────────────────────
        if has_size and all(v is not None for v in raw_sizes):
            max_ape_vals = [d[size_metric] for d in model_data]
            vmin, vmax = min(max_ape_vals), max(max_ape_vals)
            vmid = (vmin + vmax) / 2

            def _ref_size(v):
                if vmax - vmin < 1e-9:
                    return (_BUBBLE_MIN + _BUBBLE_MAX) / 2
                return _BUBBLE_MIN + (v - vmin) / (vmax - vmin) * (_BUBBLE_MAX - _BUBBLE_MIN)

            # Create reference scatter handles sized by MaxAPE (areas)
            size_handles = [
                ax.scatter([], [], s=_ref_size(v), color='#777777',
                           alpha=0.95, marker='o',
                           label=f'MaxAPE = {v:.1f}%')
                for v in [vmin, vmid, vmax]
            ]

            # Place a larger bubble-size panel at the top-right inside the axes
            # Increase spacing so the varied marker sizes do not overlap in the legend
            size_legend = ax.legend(
                handles=size_handles,
                title='Bubble size',
                title_fontsize=11,
                fontsize=10,
                loc='upper right',
                bbox_to_anchor=(0.98, 0.94),
                bbox_transform=ax.transAxes,
                framealpha=0.96,
                edgecolor='#bbbbbb',
                labelspacing=1.9,  # more vertical spacing between entries
                handletextpad=1.4,  # space between marker and text
                handlelength=2.4,  # make the marker area more visually separate
                borderpad=1.0,
            )
            ax.add_artist(size_legend)

        # ── color panel for methods (lower-left inside axes) ────────────────
        # Create a compact legend-like panel that shows each method's colour
        # at the lower-left of the plot. This replaces the previous large
        # textual summary and makes it easy to map colours to methods.
        legend_handles = [mpatches.Patch(facecolor=colors[m], edgecolor=_deepen_color(colors[m], 0.6),
                                         label=m) for m in models]
        method_legend = ax.legend(
            handles=legend_handles,
            title='Methods',
            title_fontsize=9,
            fontsize=8,
            loc='lower left',
            bbox_to_anchor=(0.02, 0.02),
            bbox_transform=ax.transAxes,
            framealpha=0.95,
            edgecolor='#cccccc',
            handlelength=1.2,
            handleheight=1.2,
            labelspacing=0.35,
        )
        ax.add_artist(method_legend)

        # Attach a tiny CV note near the lower-left if available
        cv_val = target_cvs.get(target, None)
        if cv_val is not None:
            cv_note = f"CV = {cv_val * 100:.1f}% → {y_label}"
            ax.text(0.02, 0.18, cv_note, transform=ax.transAxes,
                    fontsize=8, verticalalignment='bottom', horizontalalignment='left',
                    bbox=dict(boxstyle='round,pad=0.25', facecolor='#fffbe6',
                              edgecolor='#eeeecc', alpha=0.9), zorder=7)

        # ── top-candidates descriptive line (top-left, horizontal) ────────
        # Provide a single horizontal description line listing the top
        # candidate models and their key metrics so the plot points are not
        # obscured by multi-line boxes. Items are separated by ' | '.
        if top3:
            desc_items = []
            for i, m in enumerate(top3, 1):
                # show concise metric pair (x and y) in the description
                desc_items.append(
                    f"{i}. {m['model']} ({x_label}: {m[x_key]:.2f}, {y_label}: {m[y_key]:.3f})"
                )
            desc_line = '   |   '.join(desc_items)
            ax.text(0.02, 0.97, desc_line,
                    transform=ax.transAxes,
                    fontsize=8.5,
                    verticalalignment='top', horizontalalignment='left',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='#fffbe6',
                              edgecolor='#cccccc', alpha=0.92),
                    zorder=9)

        # ── save ─────────────────────────────────────────────────────────────
        safe = (target.replace('/', '_').replace(' ', '_')
                .replace('(', '').replace(')', ''))
        save_path = os.path.join(output_dir, f'{safe}_validation.png')
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)

        summary[target] = {
            'pareto_models': [m['model'] for m in pareto_models],
            'top3_models': [m['model'] for m in top3],
            'x_axis': x_key,
            'y_axis': y_key,
        }

    return summary


def plot_eval_results(
        all_results: Dict[str, Dict[str, Dict[str, float]]],
        output_dir: str = "./validation_plots",
        **kwargs
):
    """
    Plot all eval results
    :param all_results: dict[target][model] = metrics
    :return:
    """
    for tar, result in all_results.items():
        plot_model_validations(
            {tar: result},  # Pass single target dict
            output_dir=os.path.join(output_dir, tar.replace('/', '_')),
            **kwargs
        )
    print("save validation plots to:", output_dir)