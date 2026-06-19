from __future__ import annotations

from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT_DIR = Path("figures")
OUT_STEM = "pipeline_flowchart"
EXTRA_OUT_STEMS = ["pipeline_flowchart_clean"]


COLORS = {
    "data": "#E8EEF7",
    "gnn": "#EAF5EF",
    "target": "#FFF3D6",
    "context": "#F1ECF8",
    "classification": "#E9F7FA",
    "reconstruction": "#FDECEC",
    "llm": "#F6F6F6",
    "output": "#EDEBE7",
}


def add_box(ax, xy, width, height, title, body, color, fontsize=9.4, wrap_width=32):
    x, y = xy
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.1,
        edgecolor="#30343B",
        facecolor=color,
    )
    ax.add_patch(box)
    ax.text(
        x + width / 2,
        y + height - 0.22,
        title,
        ha="center",
        va="top",
        fontsize=10.5,
        fontweight="bold",
        color="#20242A",
    )
    body_text = "\n".join(fill(line, wrap_width) for line in body.splitlines())
    ax.text(
        x + width / 2,
        y + height - 0.58,
        body_text,
        ha="center",
        va="top",
        fontsize=fontsize,
        color="#30343B",
        linespacing=1.25,
    )
    return {
        "left": (x, y + height / 2),
        "right": (x + width, y + height / 2),
        "top": (x + width / 2, y + height),
        "bottom": (x + width / 2, y),
        "center": (x + width / 2, y + height / 2),
    }


def connect(ax, start, end, rad=0.0):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=11,
        linewidth=1.15,
        color="#3A3F46",
        shrinkA=5,
        shrinkB=5,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arrow)


def connect_polyline(ax, points):
    """Draw a clean elbow connector with the arrow only on the last segment."""
    if len(points) < 2:
        return
    for start, end in zip(points[:-2], points[1:-1]):
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color="#3A3F46",
            linewidth=1.15,
            solid_capstyle="round",
        )
    connect(ax, points[-2], points[-1])


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 9.4))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9.4)
    ax.axis("off")

    ax.text(
        0.35,
        9.02,
        "Experimental pipeline",
        fontsize=17,
        fontweight="bold",
        ha="left",
        va="center",
        color="#20242A",
    )
    ax.text(
        0.35,
        8.68,
        "From graph datasets to LLM outputs, parsed predictions, and evaluation metrics",
        fontsize=10.2,
        ha="left",
        va="center",
        color="#4B515A",
    )

    data = add_box(
        ax,
        (0.35, 6.55),
        3.0,
        1.45,
        "1. Data",
        "Load Elliptic, DGraph and IBM AML as PyG graphs.\nCreate features, labels, edges and masks.",
        COLORS["data"],
        wrap_width=34,
    )
    gnn = add_box(
        ax,
        (4.25, 6.55),
        3.0,
        1.45,
        "2. GNN models",
        "Train four GNNs:\nGCN, GAT, GIN and GraphSAGE.\nSave predictions and embeddings.",
        COLORS["gnn"],
        fontsize=8.9,
        wrap_width=32,
    )
    target = add_box(
        ax,
        (8.15, 6.55),
        3.0,
        1.45,
        "3. Targets",
        "Sample evaluation nodes from the test mask.\nSome analyses use stratified or label-balanced sampling.",
        COLORS["target"],
        wrap_width=34,
    )
    context = add_box(
        ax,
        (12.05, 6.55),
        3.0,
        1.45,
        "4. Context",
        "Extract raw features, embeddings, local graph structure, GNNExplainer artifacts and reconstruction candidate sets.",
        COLORS["context"],
        fontsize=9.0,
        wrap_width=34,
    )

    cls = add_box(
        ax,
        (1.0, 3.9),
        4.15,
        1.55,
        "5a. Classification fidelity",
        "Prompt the LLM to reproduce the GNN class.\nInputs: embedding, GNNExplainer artifacts or raw graph context.\nOutput: predicted_class.",
        COLORS["classification"],
        fontsize=9.0,
        wrap_width=42,
    )
    recon = add_box(
        ax,
        (6.0, 3.9),
        4.15,
        1.55,
        "5b. Neighbourhood reconstruction",
        "Prompt the LLM to select node IDs.\nInputs: target and candidate features or embeddings.\nOutput: selected neighbour IDs.",
        COLORS["reconstruction"],
        fontsize=9.0,
        wrap_width=42,
    )

    baselines = add_box(
        ax,
        (11.0, 3.9),
        4.15,
        1.55,
        "5c. Reconstruction baselines",
        "Evaluate random, cosine and feature-distance baselines on the same candidate sets.",
        "#F7F0E5",
        fontsize=9.0,
        wrap_width=42,
    )

    llm = add_box(
        ax,
        (3.45, 0.75),
        4.15,
        1.75,
        "6. LLM inference and parsing",
        "Run Qwen models.\nSave prompt, raw response and parsed output.\nMark unparseable class outputs as unknown.",
        COLORS["llm"],
        fontsize=9.0,
        wrap_width=42,
    )
    metrics = add_box(
        ax,
        (8.4, 0.75),
        5.6,
        1.75,
        "7. Evaluation and artifacts",
        "Classification: accuracy, precision, recall, F1, and parse rate.\nReconstruction: precision, recall, F1, Jaccard, overlap, and edit distance.\nWrite raw JSON, summary tables and plots.",
        COLORS["output"],
        fontsize=8.6,
        wrap_width=58,
    )

    connect(ax, data["right"], gnn["left"])
    connect(ax, gnn["right"], target["left"])
    connect(ax, target["right"], context["left"])
    branch_y = 5.88
    hub = (context["bottom"][0], branch_y)
    ax.plot(
        [context["bottom"][0], hub[0]],
        [context["bottom"][1], hub[1]],
        color="#3A3F46",
        linewidth=1.15,
        solid_capstyle="round",
    )
    connect_polyline(ax, [hub, (cls["top"][0], branch_y), cls["top"]])
    connect_polyline(ax, [hub, (recon["top"][0], branch_y), recon["top"]])
    connect_polyline(ax, [hub, (baselines["top"][0], branch_y), baselines["top"]])
    connect(ax, cls["bottom"], llm["top"], rad=0.08)
    connect(ax, recon["bottom"], llm["top"], rad=0.0)
    connect(ax, baselines["bottom"], metrics["top"], rad=-0.08)
    connect(ax, llm["right"], metrics["left"])

    for stem in [OUT_STEM, *EXTRA_OUT_STEMS]:
        for ext in ("pdf", "png", "svg"):
            path = OUT_DIR / f"{stem}.{ext}"
            fig.savefig(path, bbox_inches="tight", dpi=300)
            print(f"Wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
