import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUTPUT_PNG = "results/sap_architecture_diagram.png"
OUTPUT_PDF = "results/sap_architecture_diagram.pdf"


def add_box(ax, x, y, w, h, title, body):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        linewidth=1.5,
    )
    ax.add_patch(box)

    ax.text(
        x + w / 2,
        y + h * 0.68,
        title,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
    )

    ax.text(
        x + w / 2,
        y + h * 0.30,
        body,
        ha="center",
        va="center",
        fontsize=8.5,
    )


def add_arrow(ax, x1, y1, x2, y2):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=1.4,
        )
    )


def main():
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    add_box(
        ax, 0.4, 5.8, 2.0, 1.1,
        "Data Layer",
        "DataCo / IIoT\nOrder & operational data",
    )

    add_box(
        ax, 3.0, 5.8, 2.0, 1.1,
        "ML Prediction",
        "LightGBM / XGBoost\nLate-delivery probability",
    )

    add_box(
        ax, 5.6, 5.8, 2.2, 1.1,
        "Decision Layer",
        "Risk classification\nLOW / MEDIUM / HIGH",
    )

    add_box(
        ax, 8.4, 5.8, 2.2, 1.1,
        "Human-in-the-Loop",
        "Review / approve /\noverride",
    )

    add_box(
        ax, 11.1, 5.8, 2.3, 1.1,
        "SAP Workflow",
        "SAP-oriented action\n& process routing",
    )

    add_box(
        ax, 5.6, 2.9, 2.2, 1.1,
        "Economic Layer",
        "Cost-sensitive decision\n& break-even analysis",
    )

    add_box(
        ax, 8.4, 2.9, 2.2, 1.1,
        "Audit Layer",
        "Decision status\n& workflow audit trail",
    )

    add_arrow(ax, 2.4, 6.35, 3.0, 6.35)
    add_arrow(ax, 5.0, 6.35, 5.6, 6.35)
    add_arrow(ax, 7.8, 6.35, 8.4, 6.35)
    add_arrow(ax, 10.6, 6.35, 11.1, 6.35)

    add_arrow(ax, 6.7, 5.8, 6.7, 4.0)
    add_arrow(ax, 7.8, 3.45, 8.4, 3.45)
    add_arrow(ax, 9.5, 4.0, 9.5, 5.8)

    ax.add_patch(
        FancyArrowPatch(
            (12.25, 5.8),
            (12.25, 1.7),
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=1.2,
        )
    )

    ax.add_patch(
        FancyArrowPatch(
            (12.25, 1.7),
            (1.4, 1.7),
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=1.2,
        )
    )

    ax.add_patch(
        FancyArrowPatch(
            (1.4, 1.7),
            (1.4, 5.8),
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=1.2,
        )
    )

    ax.text(
        7.0,
        7.35,
        "Explainable Human-in-the-Loop AI Decision Layer "
        "for SAP ERP Integration",
        ha="center",
        fontsize=14,
        fontweight="bold",
    )

    ax.text(
        7.0,
        0.8,
        "Prediction → Risk → Decision → Human Governance → "
        "SAP Workflow → Economic Evaluation & Audit Feedback",
        ha="center",
        fontsize=10,
    )

    plt.tight_layout(pad=0.2)
    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.savefig(OUTPUT_PDF, bbox_inches="tight")

    print("Saved:")
    print(f"- {OUTPUT_PNG}")
    print(f"- {OUTPUT_PDF}")

    plt.show()


if __name__ == "__main__":
    main()