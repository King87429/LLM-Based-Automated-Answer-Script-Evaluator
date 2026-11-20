import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr

df = pd.read_csv("dataset_500.csv")

df = df.dropna(subset=["avg_human_score", "llm_score"])

human = df["avg_human_score"]
llm = df["llm_score"]

mae = mean_absolute_error(human, llm)
rmse = np.sqrt(mean_squared_error(human, llm))
corr, _ = pearsonr(human, llm)

exact_match = ((human == llm).sum() / len(df)) * 100

results = f"""
========================================
LLM GRADING EVALUATION RESULTS
========================================

Total Records: {len(df)}

Mean Absolute Error (MAE): {mae:.4f}

Root Mean Square Error (RMSE): {rmse:.4f}

Pearson Correlation: {corr:.4f}

Exact Match Percentage: {exact_match:.2f}%

========================================
"""

print(results)

with open("evaluation_results.txt", "w") as f:
    f.write(results)

print("Results saved to evaluation_results.txt")