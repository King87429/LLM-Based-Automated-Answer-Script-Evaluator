import pandas as pd
import re
from ollama import chat

# Load dataset
df = pd.read_csv("dataset.csv")

# Create columns if not present
if "llm_score" not in df.columns:
    df["llm_score"] = None

if "llm_feedback" not in df.columns:
    df["llm_feedback"] = None

total_rows = len(df)

for index, row in df.iterrows():

    # Skip already processed rows
    if pd.notna(row["llm_score"]):
        continue

    prompt = f"""
You are a strict university examiner.

QUESTION:
{row['question']}

MAXIMUM MARKS:
{row['max_score']}

RUBRIC:
{row['rubric']}

REFERENCE ANSWER:
{row['reference_answer']}

STUDENT ANSWER:
{row['student_answer']}

Evaluate according to the rubric.

Return ONLY in this exact format:

Score: X

Feedback: One sentence
"""

    try:

        response = chat(
            model="deepseek-r1:8b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        result_text = response["message"]["content"]

        # Extract score
        match = re.search(
            r"Score:\s*(\d+(\.\d+)?)",
            result_text
        )

        if match:
            score = float(match.group(1))
        else:
            score = None

        # Save result
        df.loc[index, "llm_score"] = score
        df.loc[index, "llm_feedback"] = result_text

        # Save immediately
        df.to_csv("dataset.csv", index=False)

        print(
            f"Processed {index+1}/{total_rows} | Score = {score}"
        )

    except Exception as e:

        print(
            f"Error on row {index+1}: {e}"
        )

        df.loc[index, "llm_score"] = None
        df.loc[index, "llm_feedback"] = "ERROR"

        df.to_csv("dataset.csv", index=False)

print("\nAll records processed successfully!")