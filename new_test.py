import pandas as pd
import re
from ollama import chat

# Load dataset
df = pd.read_csv("newdataset.csv")

# Create columns if they don't exist
if "llm_score" not in df.columns:
    df["llm_score"] = None

if "llm_feedback" not in df.columns:
    df["llm_feedback"] = None

total_rows = len(df)

print(f"Found {total_rows} records\n")

for index, row in df.iterrows():

    # Skip already processed rows
    if pd.notna(df.loc[index, "llm_score"]):
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

Evaluate the student answer according to the rubric.

Return ONLY:

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
            result_text,
            re.IGNORECASE
        )

        if match:
            score = float(match.group(1))
        else:
            score = None

        # Save results
        df.loc[index, "llm_score"] = score
        df.loc[index, "llm_feedback"] = result_text

        # Save immediately
        df.to_csv("newdataset.csv", index=False)

        print(f"✓ Processed Row {index+1}/{total_rows} | Score = {score}")

    except Exception as e:

        print(f"✗ Error on Row {index+1}: {e}")

        df.loc[index, "llm_score"] = None
        df.loc[index, "llm_feedback"] = f"ERROR: {e}"

        df.to_csv("newdataset.csv", index=False)

print("\nAll records processed successfully!")