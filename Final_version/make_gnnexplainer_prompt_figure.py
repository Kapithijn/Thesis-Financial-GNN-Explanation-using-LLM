from __future__ import annotations

from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT_DIRS = [Path("figures"), Path("overleaf_upload_latest_results/figures")]
OUT_STEM = "gnnexplainer_prompt_context_diagram"


COLORS = {
    "source": "#F2EFE8",
    "process": "#EAF5EF",
    "subgraph": "#EAF6FB",
    "mask": "#FDEDED",
    "template": "#F0ECFA",
    "format": "#F7F7F7",
    "eval": "#EDEBE7",
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
    body_size: float = 8.4,
    wrap_width: int = 30,
    linewidth: float = 1.05,
):
    x, y = xy
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.025,rounding_size=0.08",
        linewidth=linewidth,
        edgecolor=COLORS["line"],
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
        x + 0.22,
        y + height - 0.24,
        title,
        ha="left",
        va="top",
        fontsize=10.8,
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
        "GNNExplainer context in prompt construction",
        ha="left",
        va="center",
        fontsize=17,
        fontweight="bold",
        color=COLORS["text"],
    )
    source = add_box(
        ax,
        (0.55, 5.35),
        3.0,
        1.85,
        "GNNExplainer output",
        "Target node\nEdge importance scores\nFeature importance mask\nLocal k-hop graph\nSelected neighbors, if available",
        COLORS["source"],
        body_size=8.4,
        wrap_width=29,
    )

    rank = add_box(
        ax,
        (4.35, 5.35),
        2.8,
        1.85,
        "Rank and filter",
        "Sort edges by normalized importance\nKeep scores >= 0.70\nFallback to top 1 edge\nCap at top 5 edges",
        COLORS["process"],
        body_size=8.1,
        wrap_width=27,
    )

    format_subgraph = add_box(
        ax,
        (7.95, 5.35),
        3.1,
        1.85,
        "Compact subgraph text",
        "Nodes: target plus selected endpoints\nEdges: source -> target\nScores: raw and normalized importance",
        COLORS["subgraph"],
        body_size=8.1,
        wrap_width=30,
    )

    add_panel(ax, (0.55, 2.75), 14.9, 2.0, "A. Fidelity prompt with explainer subgraph", "#EAF6FB")
    fid_prompt = add_box(
        ax,
        (1.0, 3.08),
        3.25,
        1.2,
        "Prompt slots",
        "{explanation}: importance list\n{embedding}: target embedding\n{subgraph}: compact explainer subgraph",
        "#FFFFFF",
        body_size=8.0,
        wrap_width=34,
    )
    fid_template = add_box(
        ax,
        (5.0, 3.08),
        3.0,
        1.2,
        "Instruction",
        "Predict class 0 or 1 for the target node using the supplied GNN context.",
        COLORS["template"],
        body_size=8.0,
        wrap_width=31,
    )
    fid_output = add_box(
        ax,
        (8.75, 3.08),
        2.45,
        1.2,
        "Output format",
        "The predicted class is X",
        COLORS["format"],
        body_size=8.5,
        wrap_width=24,
    )
    fid_eval = add_box(
        ax,
        (11.95, 3.08),
        2.85,
        1.2,
        "Evaluation",
        "Compare LLM class with the GNN prediction.",
        COLORS["eval"],
        body_size=8.0,
        wrap_width=29,
    )

    add_panel(ax, (0.55, 0.45), 14.9, 2.0, "B. Reconstruction prompt with explainer information", "#FDEDED")
    recon_mask = add_box(
        ax,
        (1.0, 0.78),
        3.25,
        1.2,
        "Explainer block",
        "Feature/importance mask is formatted as explanation text.",
        COLORS["mask"],
        body_size=8.0,
        wrap_width=32,
    )
    recon_prompt = add_box(
        ax,
        (5.0, 0.78),
        3.0,
        1.2,
        "Prompt slots",
        "Explanation + target embedding/features + candidate context + candidate IDs.",
        "#FFFFFF",
        body_size=8.0,
        wrap_width=32,
    )
    recon_output = add_box(
        ax,
        (8.75, 0.78),
        2.45,
        1.2,
        "Output format",
        '{"selected_neighbors": [...]}',
        COLORS["format"],
        body_size=8.0,
        wrap_width=27,
    )
    recon_eval = add_box(
        ax,
        (11.95, 0.78),
        2.85,
        1.2,
        "Evaluation",
        "Compare selected IDs with true one-hop neighbors.",
        COLORS["eval"],
        body_size=8.0,
        wrap_width=29,
    )

    add_arrow(ax, source["right"], rank["left"])
    add_arrow(ax, rank["right"], format_subgraph["left"])
    for left, right in [
        (fid_prompt, fid_template),
        (fid_template, fid_output),
        (fid_output, fid_eval),
        (recon_mask, recon_prompt),
        (recon_prompt, recon_output),
        (recon_output, recon_eval),
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
