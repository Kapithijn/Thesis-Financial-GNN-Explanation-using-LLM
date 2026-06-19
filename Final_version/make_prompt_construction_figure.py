from __future__ import annotations

from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT_DIRS = [Path("figures"), Path("overleaf_upload_latest_results/figures")]
OUT_STEM = "prompt_construction_diagram"


COLORS = {
    "fidelity_panel": "#EAF6FB",
    "reconstruction_panel": "#FDEDED",
    "input": "#FFFFFF",
    "template": "#F0ECFA",
    "format": "#F7F7F7",
    "eval": "#EAF6EF",
    "line": "#30343B",
    "text": "#20242A",
    "muted": "#555D66",
}


def wrap_text(text: str, width: int) -> str:
    return "\n".join(fill(line, width=width) for line in text.splitlines())


def add_box(
    ax,
    xy: tuple[float, float],
    width: float,
    height: float,
    title: str,
    body: str,
    color: str,
    *,
    title_size: float = 10.0,
    body_size: float = 8.5,
    wrap_width: int = 28,
    edge: str = COLORS["line"],
    linewidth: float = 1.05,
):
    x, y = xy
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.025,rounding_size=0.08",
        linewidth=linewidth,
        edgecolor=edge,
        facecolor=color,
    )
    ax.add_patch(box)
    ax.text(
        x + width / 2,
        y + height - 0.18,
        title,
        ha="center",
        va="top",
        fontsize=title_size,
        fontweight="bold",
        color=COLORS["text"],
    )
    ax.text(
        x + width / 2,
        y + height - 0.52,
        wrap_text(body, wrap_width),
        ha="center",
        va="top",
        fontsize=body_size,
        color=COLORS["muted"],
        linespacing=1.22,
    )
    return {
        "left": (x, y + height / 2),
        "right": (x + width, y + height / 2),
        "top": (x + width / 2, y + height),
        "bottom": (x + width / 2, y),
        "center": (x + width / 2, y + height / 2),
    }


def add_panel(ax, xy, width, height, title, color):
    x, y = xy
    panel = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=0.9,
        edgecolor="#B8C0CC",
        facecolor=color,
    )
    ax.add_patch(panel)
    ax.text(
        x + 0.24,
        y + height - 0.25,
        title,
        ha="left",
        va="top",
        fontsize=11.0,
        fontweight="bold",
        color=COLORS["text"],
    )


def add_arrow(ax, start, end, *, rad: float = 0.0):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=11,
        linewidth=1.1,
        color=COLORS["line"],
        shrinkA=5,
        shrinkB=5,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arrow)


def add_tag(ax, x: float, y: float, text: str, color: str):
    box = FancyBboxPatch(
        (x, y),
        1.62,
        0.28,
        boxstyle="round,pad=0.015,rounding_size=0.07",
        linewidth=0.6,
        edgecolor="#9EA7B3",
        facecolor=color,
    )
    ax.add_patch(box)
    ax.text(
        x + 0.81,
        y + 0.14,
        text,
        ha="center",
        va="center",
        fontsize=7.2,
        color=COLORS["text"],
    )


def main() -> None:
    for out_dir in OUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 8.8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8.8)
    ax.axis("off")

    ax.text(
        0.45,
        8.45,
        "Prompt construction for LLM experiments",
        ha="left",
        va="center",
        fontsize=17,
        fontweight="bold",
        color=COLORS["text"],
    )
    add_panel(ax, (0.55, 4.75), 14.9, 2.65, "A. Fidelity / classification prompt", COLORS["fidelity_panel"])
    add_panel(ax, (0.55, 1.35), 14.9, 2.65, "B. Neighbourhood reconstruction prompt", COLORS["reconstruction_panel"])

    fidelity_context = add_box(
        ax,
        (0.88, 5.05),
        2.45,
        1.55,
        "Context fields",
        "Explanation mask\nTarget embedding\nSubgraph or raw graph context",
        COLORS["input"],
        body_size=8.3,
        wrap_width=26,
    )
    fidelity_formatting = add_box(
        ax,
        (3.75, 5.05),
        2.45,
        1.55,
        "Formatting",
        "Convert arrays, masks, and graph context into compact text blocks",
        COLORS["format"],
        body_size=8.2,
        wrap_width=25,
    )
    fidelity_template = add_box(
        ax,
        (6.62, 5.05),
        2.45,
        1.55,
        "Template",
        "Task instruction\nClass definitions\nFew-shot examples\nPlaceholder insertion",
        COLORS["template"],
        body_size=8.0,
        wrap_width=25,
    )
    fidelity_format = add_box(
        ax,
        (9.49, 5.05),
        2.25,
        1.55,
        "Required answer",
        "The predicted class is X\nX in {0, 1}",
        COLORS["format"],
        body_size=8.7,
        wrap_width=25,
    )
    fidelity_eval = add_box(
        ax,
        (12.29, 5.05),
        2.75,
        1.55,
        "Evaluation",
        "Compare LLM class with the GNN prediction\nReport fidelity metrics",
        COLORS["eval"],
        body_size=8.1,
        wrap_width=24,
    )

    add_tag(ax, 10.03, 6.75, "embedding", "#D9EEF7")
    add_tag(ax, 11.75, 6.75, "+ explainer", "#D9EEF7")
    add_tag(ax, 13.47, 6.75, "raw graph", "#D9EEF7")

    recon_context = add_box(
        ax,
        (0.88, 1.65),
        2.45,
        1.55,
        "Context fields",
        "Target embedding/features\nCandidate context\nCandidate node IDs",
        COLORS["input"],
        body_size=8.3,
        wrap_width=25,
    )
    recon_formatting = add_box(
        ax,
        (3.75, 1.65),
        2.45,
        1.55,
        "Formatting",
        "Format target and candidate vectors; optionally add explainer text",
        COLORS["format"],
        body_size=8.2,
        wrap_width=25,
    )
    recon_template = add_box(
        ax,
        (6.62, 1.65),
        2.45,
        1.55,
        "Template",
        "Strict JSON instruction\nSelect direct 1-hop neighbors\nUse candidate IDs only",
        COLORS["template"],
        body_size=8.2,
        wrap_width=25,
    )
    recon_format = add_box(
        ax,
        (9.49, 1.65),
        2.25,
        1.55,
        "Required answer",
        '{"selected_neighbors": [...],\n "confidence": 0.0}',
        COLORS["format"],
        body_size=8.0,
        wrap_width=26,
    )
    recon_eval = add_box(
        ax,
        (12.29, 1.65),
        2.75,
        1.55,
        "Evaluation",
        "Compare selected IDs with true 1-hop neighbors\nReport F1 and Jaccard",
        COLORS["eval"],
        body_size=8.1,
        wrap_width=24,
    )

    add_tag(ax, 10.03, 3.35, "GNN context", "#F7DCDC")
    add_tag(ax, 11.75, 3.35, "+ explainer", "#F7DCDC")
    add_tag(ax, 13.47, 3.35, "no GNN", "#F7DCDC")

    for left, right in [
        (fidelity_context, fidelity_formatting),
        (fidelity_formatting, fidelity_template),
        (fidelity_template, fidelity_format),
        (fidelity_format, fidelity_eval),
        (recon_context, recon_formatting),
        (recon_formatting, recon_template),
        (recon_template, recon_format),
        (recon_format, recon_eval),
    ]:
        add_arrow(ax, left["right"], right["left"])

    for out_dir in OUT_DIRS:
        for ext in ("pdf", "png", "svg"):
            path = out_dir / f"{OUT_STEM}.{ext}"
            fig.savefig(path, bbox_inches="tight", dpi=300)
            print(f"Wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
