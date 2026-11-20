import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr

df = pd.read_csv("dataset_500.csv")

human = df["avg_human_score"]
llm = df["llm_score"]

mae = mean_absolute_error(human, llm)
rmse = np.sqrt(mean_squared_error(human, llm))
corr, _ = pearsonr(human, llm)

exact_match = ((human == llm).sum()/len(df))*100

analysis = f"""
LLM GRADING ANALYSIS REPORT

Dataset Size: {len(df)}

MAE: {mae:.4f}

RMSE: {rmse:.4f}

Pearson Correlation: {corr:.4f}

Exact Match Percentage: {exact_match:.2f}%

INTERPRETATION

1. Lower MAE indicates closer agreement between
   human and LLM grading.

2. Lower RMSE indicates fewer large grading errors.

3. Pearson Correlation closer to 1 indicates
   stronger agreement between human and LLM.

4. Exact Match Percentage shows how often
   the LLM assigned exactly the same score
   as human evaluators.
"""

with open("analysis_report.txt", "w") as f:
    f.write(analysis)

print(analysis)