import pandas as pd
import matplotlib.pyplot as plt

# Set font sizes for better readability
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12
})

df = pd.read_csv("F:\Extension\Result\Result_summary - python_sample_with_scores_side.csv")

fig, axes = plt.subplots(1, 2, figsize=(10,4))

# BLEU-4
axes[0].boxplot([
    df["phi_nocontext_BLEU_4_score"],
    df["gpt_BLEU_4_score"],
    df["phi_BLEU_4_score"]
], showmeans=True)

axes[0].set_xticklabels([
    "Phi\n(No Ctx)",
    "GPT-5.4",
    "Full RAG"
])

axes[0].set_ylabel("BLEU-4")
axes[0].set_title("BLEU-4")

# ROUGE-L
axes[1].boxplot([
    df["phi_nocontext_ROUGE_L_score"],
    df["gpt_ROUGE_L_score"],
    df["phi_ROUGE_L_score"]
], showmeans=True)

axes[1].set_xticklabels([
    "Phi\n(No Ctx)",
    "GPT-5.4",
    "Full RAG"
])

axes[1].set_ylabel("ROUGE-L")
axes[1].set_title("ROUGE-L")

for ax in axes:
    ax.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig("boxplots_bleu_rouge.pdf", dpi=300)
plt.savefig("boxplots_bleu_rouge.png", dpi=300)

plt.show()