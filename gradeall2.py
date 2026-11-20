import sys
import json
import re
import pandas as pd
from ollama import chat

# =================================================
# 1. SETTINGS  (only change these)
# =================================================
FILE_NAME = "dataset_500.csv"        # <-- your new CSV name (or run: python grade_with_llm.py yourfile.csv)
MODEL = "deepseek-r1:8b"
MAX_RETRIES = 3
START_FRESH = True               # True  = grade from row 1, ignoring any old llm_score/llm_feedback
                                 # False = resume (skip rows already graded)

if len(sys.argv) > 1:
    FILE_NAME = sys.argv[1]

SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "number"},
        "feedback": {"type": "string"},
    },
    "required": ["score", "feedback"],
}

# =================================================
# 2. LOAD + VALIDATE DATASET
# =================================================
df = pd.read_csv(FILE_NAME)

required = ["question", "max_score", "rubric", "reference_answer", "student_answer"]
missing = [c for c in required if c not in df.columns]
if missing:
    sys.exit(f"\nERROR: these columns are missing in {FILE_NAME}: {missing}\n"
             f"Columns found: {list(df.columns)}")

df["max_score"] = pd.to_numeric(df["max_score"], errors="coerce")

if START_FRESH or "llm_score" not in df.columns:
    df["llm_score"] = float("nan")
if START_FRESH or "llm_feedback" not in df.columns:
    df["llm_feedback"] = ""

df["llm_score"] = pd.to_numeric(df["llm_score"], errors="coerce")
df["llm_feedback"] = df["llm_feedback"].fillna("").astype(str).astype(object)

total_rows = len(df)
print(f"\nFile: {FILE_NAME}")
print(f"Total records found: {total_rows}")
print(f"Mode: {'START FROM ROW 1' if START_FRESH else 'RESUME'}")
print("Starting LLM grading...\n")


# =================================================
# 3. HELPERS
# =================================================
def build_prompt(row):
    return f"""You are an experienced and strict university examiner.

Evaluate the student's answer ONLY according to the question, rubric and reference answer.

QUESTION:
{row['question']}

MAXIMUM MARKS: {row['max_score']}

RUBRIC:
{row['rubric']}

REFERENCE ANSWER:
{row['reference_answer']}

STUDENT ANSWER:
{row['student_answer']}

GRADING RULES:
1. Follow the rubric strictly and add up the marks for each point covered.
2. Do not award marks for incorrect information.
3. Give partial marks when only some required concepts are covered.
4. Do not require the student's wording to match the reference answer exactly.
5. The score must be a number between 0 and {row['max_score']}.
6. The feedback must be 1-2 sentences explaining why marks were awarded or deducted.

Respond ONLY with JSON in this form:
{{"score": <number>, "feedback": "<short explanation>"}}
"""


def call_model(prompt):
    kwargs = dict(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        format=SCHEMA,
        options={"temperature": 0, "num_ctx": 8192, "num_predict": 1024},
    )
    try:
        response = chat(think=False, **kwargs)   # disables <think> output
    except TypeError:
        response = chat(**kwargs)                # older ollama library
    return response["message"]["content"] or ""


def parse_result(text):
    """Return (score, feedback); either can be None if not found."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"```(?:json)?", "", text).strip()

    block = re.search(r"\{.*\}", text, re.DOTALL)
    if block:
        try:
            data = json.loads(block.group(0))
            return float(data["score"]), str(data.get("feedback", "")).strip()
        except Exception:
            pass

    score_m = re.search(r'"?score"?\s*[:=]\s*"?(\d+(?:\.\d+)?)', text, re.IGNORECASE)
    fb_m = re.search(r'"?feedback"?\s*[:=]\s*"(.*?)"\s*[,}]', text, re.IGNORECASE | re.DOTALL)
    if not fb_m:
        fb_m = re.search(r'"?feedback"?\s*[:=]\s*"?(.*?)"?\s*}?\s*$', text, re.IGNORECASE | re.DOTALL)

    score = float(score_m.group(1)) if score_m else None
    feedback = fb_m.group(1).strip(' "\n') if fb_m else None
    return score, feedback


def grade_row(row):
    """Retry until BOTH score and feedback are obtained."""
    prompt = build_prompt(row)
    last_error = "unknown error"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw = call_model(prompt)
            score, feedback = parse_result(raw)

            if score is None:
                last_error = f"no score in output: {raw[:150]!r}"
            elif not feedback:
                last_error = f"no feedback in output: {raw[:150]!r}"
            else:
                return max(0.0, min(score, float(row["max_score"]))), feedback
        except Exception as e:
            last_error = str(e)

        print(f"   attempt {attempt}/{MAX_RETRIES} failed: {last_error}")

    raise RuntimeError(last_error)


# =================================================
# 4. GRADE EVERY ROW, STARTING FROM ROW 1
# =================================================
for index, row in df.iterrows():

    already_done = (
        pd.notna(row["llm_score"])
        and str(row["llm_feedback"]).strip() != ""
        and not str(row["llm_feedback"]).startswith("ERROR")
    )
    if already_done and not START_FRESH:
        print(f"Skipping Row {index + 1} - already graded")
        continue

    try:
        score, feedback = grade_row(row)
        df.at[index, "llm_score"] = score
        df.at[index, "llm_feedback"] = feedback
        df.to_csv(FILE_NAME, index=False)        # save after every row

        human = row["avg_human_score"] if "avg_human_score" in df.columns else "n/a"
        print(f"Processed {index + 1}/{total_rows} | Human = {human} | LLM = {score}")
        print(f"Feedback: {feedback}\n")

    except Exception as e:
        print(f"ERROR on Row {index + 1}: {e}\n")
        df.at[index, "llm_score"] = float("nan")
        df.at[index, "llm_feedback"] = f"ERROR: {e}"
        df.to_csv(FILE_NAME, index=False)

print("\n====================================")
print("ALL RECORDS PROCESSED")
print(f"Results saved in {FILE_NAME}")
print("====================================")