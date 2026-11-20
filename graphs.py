import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("dataset_500.csv")

human = df["avg_human_score"]
llm = df["llm_score"]

# -----------------------------------
# 1. Scatter Plot
# -----------------------------------

plt.figure(figsize=(8,6))
plt.scatter(human, llm)
plt.xlabel("Human Score")
plt.ylabel("LLM Score")
plt.title("Human vs LLM Scores")
plt.grid(True)

plt.savefig("scatter_plot.png")
plt.close()

# -----------------------------------
# 2. Score Distribution
# -----------------------------------

plt.figure(figsize=(8,6))

plt.hist(human, alpha=0.5, label="Human")
plt.hist(llm, alpha=0.5, label="LLM")

plt.xlabel("Score")
plt.ylabel("Frequency")
plt.title("Score Distribution")
plt.legend()

plt.savefig("score_distribution.png")
plt.close()

# -----------------------------------
# 3. Error Distribution
# -----------------------------------

error = human - llm

plt.figure(figsize=(8,6))

plt.hist(error)

plt.xlabel("Human - LLM")
plt.ylabel("Frequency")
plt.title("Error Distribution")

plt.savefig("error_distribution.png")
plt.close()

# -----------------------------------
# 4. Average Score Comparison
# -----------------------------------

plt.figure(figsize=(6,5))

avg_human = human.mean()
avg_llm = llm.mean()

plt.bar(
    ["Human", "LLM"],
    [avg_human, avg_llm]
)

plt.ylabel("Average Score")
plt.title("Average Score Comparison")

plt.savefig("average_score_comparison.png")
plt.close()

print("All graphs saved successfully!")