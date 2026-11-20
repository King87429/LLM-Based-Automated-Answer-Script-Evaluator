import sys
import time
import shutil
import json
import re
import pandas as pd
from ollama import Client

# =================================================
# 1. SETTINGS  (only change these)
# =================================================
FILE_NAME = "dataset_500.csv"        # <-- your new CSV name (or run: python grade_with_llm.py yourfile.csv)
MODEL = "deepseek-r1:8b"
REQUEST_TIMEOUT = 150            # seconds to wait for ONE model reply before giving up and trying the next attempt
MAX_RETRIES = 6                  # attempts per row in each pass (each attempt uses a different method)
REPAIR_PASSES = 3                # after the main run, re-try rows that still show ERROR this many times
ONLY_ERRORS = True               # True = ONLY re-grade rows whose llm_feedback starts with ERROR (nothing else is touched)
START_FRESH = False              # False = resume (skips rows already graded; a brand-new CSV still starts at row 1)
                                 # True  = regrade everything from row 1, overwriting old results

if len(sys.argv) > 1:
    FILE_NAME = sys.argv[1]

if ONLY_ERRORS:
    START_FRESH = False          # never wipe existing results in this mode

client = Client(timeout=REQUEST_TIMEOUT)

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
shutil.copy(FILE_NAME, FILE_NAME + ".backup.csv")   # safety copy before anything is changed

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
print(f"Mode: {'ONLY ERROR ROWS' if ONLY_ERRORS else ('START FROM ROW 1' if START_FRESH else 'RESUME')}")
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


def build_short_prompt(row):
    """Shorter prompt used on later attempts (less for the model to get stuck on)."""
    return f"""Grade this student answer out of {row['max_score']}.

Question: {row['question']}
Rubric: {row['rubric']}
Reference answer: {row['reference_answer']}
Student answer: {row['student_answer']}

Reply with ONLY this JSON and nothing else:
{{"score": <number from 0 to {row['max_score']}>, "feedback": "<one or two sentences>"}}"""


def _get(obj, name):
    """Read a field from an ollama response object or a dict."""
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


# Each attempt is DIFFERENT. (With temperature 0 the same request fails the same way every time.)
# (think, format, temperature, num_predict, short_prompt)
ATTEMPTS = [
    (True,  None,     0.0, 4096,  False),   # 1: normal thinking mode
    (False, SCHEMA,   0.0, 1024,  False),   # 2: JSON schema, no thinking
    (False, "json",   0.3, 1024,  False),   # 3: JSON mode, slight randomness
    (True,  None,     0.5, 8192,  True),    # 4: thinking, short prompt, big token limit
    (False, None,     0.3, 1024,  True),    # 5: no thinking, no format, short prompt
    (False, SCHEMA,   0.7, 1024,  True),    # 6: schema, short prompt, more randomness
]


def call_model(row, attempt):
    think, fmt, temp, n_predict, short = ATTEMPTS[(attempt - 1) % len(ATTEMPTS)]
    prompt = build_short_prompt(row) if short else build_prompt(row)
    messages = [{"role": "user", "content": prompt}]
    options = {"temperature": temp, "num_ctx": 8192, "num_predict": n_predict,
               "seed": attempt * 7 + 1}

    kwargs = dict(model=MODEL, messages=messages, options=options)
    if fmt is not None:
        kwargs["format"] = fmt
    try:
        response = client.chat(think=think, **kwargs)
    except TypeError:
        response = client.chat(**kwargs)        # older ollama library without `think`

    msg = _get(response, "message")
    content = _get(msg, "content") or ""
    thinking = _get(msg, "thinking") or ""

    # If the final answer is empty, the JSON may be inside the thinking text
    if not content.strip() and thinking.strip():
        content = thinking
    return content


def parse_result(text):
    """Return (score, feedback); either can be None if not found."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"```(?:json)?", "", text).strip()

    for block in reversed(re.findall(r"\{[^{}]*\}", text, re.DOTALL)):
        try:
            data = json.loads(block)
            return float(data["score"]), str(data.get("feedback", "")).strip()
        except Exception:
            continue

    score_m = re.search(r'"?score"?\s*[:=]\s*"?(\d+(?:\.\d+)?)', text, re.IGNORECASE)
    fb_m = re.search(r'"?feedback"?\s*[:=]\s*"(.*?)"\s*[,}]', text, re.IGNORECASE | re.DOTALL)
    if not fb_m:
        fb_m = re.search(r'"?feedback"?\s*[:=]\s*"?(.*?)"?\s*}?\s*$', text, re.IGNORECASE | re.DOTALL)

    score = float(score_m.group(1)) if score_m else None
    feedback = fb_m.group(1).strip(' "\n') if fb_m else None
    return score, feedback


def grade_row(row):
    """Try different methods until we get BOTH a score and feedback."""
    last_error = "unknown error"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"   Row {row.name + 1}: attempt {attempt}/{MAX_RETRIES} ...", flush=True)
            raw = call_model(row, attempt)
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
        time.sleep(2)

    raise RuntimeError(last_error)


# =================================================
# 4. GRADE EVERY ROW, THEN RE-TRY ANY ERROR ROWS
# =================================================
def is_done(i):
    fb = str(df.at[i, "llm_feedback"]).strip()
    return pd.notna(df.at[i, "llm_score"]) and fb != "" and not fb.startswith("ERROR")


def process(indices, label):
    for index in indices:
        row = df.loc[index]
        try:
            score, feedback = grade_row(row)
            df.at[index, "llm_score"] = score
            df.at[index, "llm_feedback"] = feedback
            df.to_csv(FILE_NAME, index=False)        # save after every row

            human = row["avg_human_score"] if "avg_human_score" in df.columns else "n/a"
            print(f"{label} {index + 1}/{total_rows} | Human = {human} | LLM = {score}")
            print(f"Feedback: {feedback}\n")
        except Exception as e:
            print(f"ERROR on Row {index + 1}: {e}\n")
            df.at[index, "llm_score"] = float("nan")
            df.at[index, "llm_feedback"] = f"ERROR: {e}"
            df.to_csv(FILE_NAME, index=False)


# main pass
if ONLY_ERRORS:
    todo = [i for i in df.index if str(df.at[i, "llm_feedback"]).startswith("ERROR")]
else:
    todo = [i for i in df.index if START_FRESH or not is_done(i)]
print(f"Rows to grade: {len(todo)} (leaving {total_rows - len(todo)} other rows untouched)")
if todo:
    print(f"Row numbers: {[i + 1 for i in todo]}\n")
try:
    process(todo, "Processed")
except KeyboardInterrupt:
    df.to_csv(FILE_NAME, index=False)
    print("\nStopped by you (Ctrl+C). Progress is saved - run the script again to continue.")
    sys.exit(0)

# repair passes for rows that still have errors
for p in range(1, REPAIR_PASSES + 1):
    failed = [i for i in todo if not is_done(i)]
    if not failed:
        break
    print(f"\n--- Repair pass {p}/{REPAIR_PASSES}: {len(failed)} rows still failing ---\n")
    try:
        process(failed, "Repaired")
    except KeyboardInterrupt:
        df.to_csv(FILE_NAME, index=False)
        print("\nStopped by you (Ctrl+C). Progress is saved - run the script again to continue.")
        sys.exit(0)

still_failed = [i + 1 for i in todo if not is_done(i)]

print("\n====================================")
print("ALL RECORDS PROCESSED")
print(f"Results saved in {FILE_NAME}")
if still_failed:
    print(f"Rows still failing ({len(still_failed)}): {still_failed}")
    print("Run the script again to retry only these rows.")
else:
    print("All selected rows now have a score and feedback.")
print("====================================")