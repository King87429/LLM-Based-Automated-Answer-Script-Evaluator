"""
grader.py - sends the question, max marks, rubric, optional reference answer and the
(teacher-corrected) student answer to a model through Ollama.

Two modes:
  local  -> Ollama on this computer (http://localhost:11434), model deepseek-r1:8b
  online -> Ollama Cloud API (https://ollama.com), key from OLLAMA_API_KEY
            (environment variable or Streamlit Secrets)

Uses the SAME prompt and the SAME retry strategy as the research pipeline
(gradeall2.py / gradeerror.py) so dashboard marks are consistent with the experiment.
"""
import json
import os
import re
import time

from ollama import Client

LOCAL_MODEL = "deepseek-r1:8b"
MODEL_NAME = LOCAL_MODEL     # kept for older scripts (gradeall2.py etc.) - always the local model

# Preferred cloud model. Override WITHOUT editing code by adding  OLLAMA_CLOUD_MODEL = "..."
# to Streamlit Secrets (or as an environment variable). If the name is not in the hosted
# catalogue, _resolve_cloud_model() automatically falls back to the first available one below.
CLOUD_MODEL = "gpt-oss:120b"
CLOUD_FALLBACKS = ["gpt-oss:120b", "gpt-oss:20b", "deepseek-v4-flash", "kimi-k2.6", "glm-5"]

LOCAL_HOST = "http://localhost:11434"
CLOUD_HOST = "https://ollama.com"
REQUEST_TIMEOUT = 150        # seconds to wait for ONE model reply before trying the next attempt
ACTIVE_MODE = "local"        # "local" or "online" - set with set_mode()

SCORE_SCHEMA = {
    "type": "object",
    "properties": {"score": {"type": "number"}, "feedback": {"type": "string"}},
    "required": ["score", "feedback"],
}

RUBRIC_SCHEMA = {
    "type": "object",
    "properties": {
        "key_points": {"type": "array", "items": {"type": "string"}},
        "reference_answer": {"type": "string"},
    },
    "required": ["key_points", "reference_answer"],
}

# (think, format, temperature, num_predict, short_prompt)
# Attempt 1 is exactly the configuration used for the research dataset grading.
ATTEMPTS = [
    (False, SCORE_SCHEMA, 0.0, 1024, False),
    (True,  None,         0.0, 4096, False),
    (False, "json",       0.3, 1024, False),
    (True,  None,         0.5, 8192, True),
    (False, None,         0.3, 1024, True),
    (False, SCORE_SCHEMA, 0.7, 1024, True),
]


class GradingError(Exception):
    pass


_client = None
_cloud_model_cache = None
_ollama_blocked_until = 0.0      # when Ollama Cloud refuses (no credits), use Gemini until this time
_gemini_used = ""                # name of the Gemini model that last answered


def _secret(name):
    """Read a setting from the environment, then from Streamlit Secrets ('' if not found)."""
    val = os.getenv(name)
    if val:
        return val
    try:
        import streamlit as st
        return str(st.secrets.get(name, "") or "")
    except Exception:
        return ""


def get_api_key():
    return _secret("OLLAMA_API_KEY").strip()


def set_mode(mode):
    """Switch between "local" and "online". Safe to call on every Streamlit rerun."""
    global ACTIVE_MODE, _client
    if mode not in ("local", "online"):
        raise ValueError("Mode must be 'local' or 'online'")
    if mode != ACTIVE_MODE:
        ACTIVE_MODE = mode
        _client = None


def _cloud_client(timeout=REQUEST_TIMEOUT):
    api_key = get_api_key()
    if not api_key:
        raise GradingError("OLLAMA_API_KEY is missing. Add it in Streamlit Secrets (or set it as an environment variable).")
    return Client(host=CLOUD_HOST, headers={"Authorization": f"Bearer {api_key}"}, timeout=timeout)


def _get_client():
    global _client
    if _client is None:
        if ACTIVE_MODE == "local":
            _client = Client(host=LOCAL_HOST, timeout=REQUEST_TIMEOUT)
        else:
            _client = _cloud_client()
    return _client


def _resolve_cloud_model():
    """Use the preferred cloud model if the hosted catalogue has it, else the first fallback that exists."""
    global _cloud_model_cache
    if _cloud_model_cache:
        return _cloud_model_cache
    wanted = _secret("OLLAMA_CLOUD_MODEL").strip() or CLOUD_MODEL
    try:
        resp = _cloud_client(timeout=15).list()
        names = []
        for m in (_field(resp, "models") or []):
            n = _field(m, "model") or _field(m, "name")
            if n:
                names.append(str(n))
        for cand in [wanted] + CLOUD_FALLBACKS:
            if cand in names:
                wanted = cand
                break
        _cloud_model_cache = wanted            # cache only after a successful catalogue lookup
    except Exception:
        pass                                   # listing failed - just try the preferred name (not cached)
    return wanted


def _gemini_key():
    return _secret("GEMINI_API_KEY").strip()


def get_model_name():
    if ACTIVE_MODE == "local":
        return LOCAL_MODEL
    if _gemini_key() and (not get_api_key() or time.time() < _ollama_blocked_until):
        return _gemini_used or "Google Gemini"
    return _resolve_cloud_model()


def _field(obj, name):
    """Read a field from an ollama response object or a plain dict."""
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


# -------------------------------------------------------------------------
# Ollama status
# -------------------------------------------------------------------------
def check_ollama():
    """Return (ok, message). ok is True only if Ollama runs AND the model is installed."""
    try:
        resp = Client(timeout=5).list()
        models = _field(resp, "models") or []
        names = []
        for m in models:
            n = _field(m, "model") or _field(m, "name")
            if n:
                names.append(str(n))
    except Exception:
        return False, "Ollama is not running. Start the Ollama app, then refresh this page."

    if any(n == MODEL_NAME or n.startswith(MODEL_NAME) for n in names):
        return True, f"Ollama is running and {MODEL_NAME} is installed."
    return False, f"Ollama is running but {MODEL_NAME} is not installed. Run: ollama pull {MODEL_NAME}"


def check_cloud():
    """Return (ok, message) for online mode. Makes no network call - it only checks that a key exists."""
    if get_api_key() and _gemini_key():
        return True, "Online mode: Ollama Cloud key found (Gemini key found as backup)."
    if get_api_key():
        return True, "Online mode: Ollama Cloud API key found."
    if _gemini_key():
        return True, "Online mode: Google Gemini key found."
    return False, "Online mode needs OLLAMA_API_KEY (or GEMINI_API_KEY) in Streamlit Secrets."


# -------------------------------------------------------------------------
# Rubric helpers
# -------------------------------------------------------------------------
def _num(x):
    return f"{round(float(x), 2):g}"


def default_rubric(max_marks):
    """General rubric template. The teacher should edit it to match the question."""
    each = _num(float(max_marks) / 5)
    return (
        f"Score out of {_num(max_marks)}. Award ~{each} points for each of the following criteria: "
        "1) Relevance - the answer addresses the question that was asked. "
        "2) Conceptual accuracy - the statements made are factually correct. "
        "3) Coverage - the key points expected for this question are covered. "
        "4) Explanation - concepts are explained with reasons, examples or applications. "
        "5) Clarity and organisation - the answer is well structured and easy to follow. "
        "Deduct for incorrect statements."
    )


def _rubric_from_points(points, max_marks):
    each = _num(float(max_marks) / len(points))
    numbered = " ".join(f"{i}) {p.strip().rstrip('.')}." for i, p in enumerate(points, 1))
    return (
        f"Score out of {_num(max_marks)}. Award ~{each} points for each of the following points "
        f"the answer correctly covers: {numbered} "
        "Deduct for incorrect statements; consider overall clarity and completeness."
    )


# -------------------------------------------------------------------------
# Prompts
# -------------------------------------------------------------------------
def _reference_text(reference):
    ref = (reference or "").strip()
    return ref if ref else "Not provided. Judge correctness using the rubric and your subject knowledge."


def build_prompt(question, max_marks, rubric, reference, answer):
    return f"""You are an experienced and strict university examiner.

Evaluate the student's answer ONLY according to the question, rubric and reference answer.

QUESTION:
{question}

MAXIMUM MARKS: {_num(max_marks)}

RUBRIC:
{rubric}

REFERENCE ANSWER:
{_reference_text(reference)}

STUDENT ANSWER:
{answer}

GRADING RULES:
1. Follow the rubric strictly and add up the marks for each point covered.
2. Do not award marks for incorrect information.
3. Give partial marks when only some required concepts are covered.
4. Do not require the student's wording to match the reference answer exactly.
5. The score must be a number between 0 and {_num(max_marks)}.
6. The feedback must be 1-2 sentences explaining why marks were awarded or deducted.
7. The answer was read from handwriting by OCR, so ignore small spelling or OCR mistakes.

Respond ONLY with JSON in this form:
{{"score": <number>, "feedback": "<short explanation>"}}
"""


def build_short_prompt(question, max_marks, rubric, reference, answer):
    return f"""Grade this student answer out of {_num(max_marks)}.

Question: {question}
Rubric: {rubric}
Reference answer: {_reference_text(reference)}
Student answer: {answer}

Reply with ONLY this JSON and nothing else:
{{"score": <number from 0 to {_num(max_marks)}>, "feedback": "<one or two sentences>"}}"""


# -------------------------------------------------------------------------
# Model call + parsing
# -------------------------------------------------------------------------
def _is_payment_error(e):
    t = str(e).lower()
    return "402" in t or "usage credits" in t or "subscribe" in t or "usage limit" in t or "429" in t


def _gemini_generate(prompt, fmt, temperature, num_predict, seed):
    """Text generation with Google Gemini (free key) - used when Ollama Cloud needs credits."""
    global _gemini_used
    import requests
    import ocr                                    # reuses the live list of current Gemini models
    cfg = {"temperature": temperature, "maxOutputTokens": max(num_predict, 4096), "seed": int(seed)}
    if fmt is not None:
        cfg["responseMimeType"] = "application/json"
    payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": cfg}
    errors = []
    for model in ocr._gemini_models():
        try:
            r = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                              headers={"x-goog-api-key": _gemini_key()}, json=payload, timeout=REQUEST_TIMEOUT)
            if r.status_code != 200:
                errors.append(f"{model}: HTTP {r.status_code} {r.text[:80]}")
                continue
            parts = r.json()["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
            if text.strip():
                _gemini_used = model
                return text
            errors.append(f"{model}: empty reply")
        except Exception as e:
            errors.append(f"{model}: {str(e)[:80]}")
    raise GradingError("Gemini could not answer. " + "; ".join(errors))


def _call_model(prompt, think, fmt, temperature, num_predict, seed):
    global _ollama_blocked_until
    if ACTIVE_MODE == "online" and _gemini_key():
        if not get_api_key() or time.time() < _ollama_blocked_until:
            return _gemini_generate(prompt, fmt, temperature, num_predict, seed)
    kwargs = dict(
        model=get_model_name(),
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": temperature, "num_ctx": 8192, "num_predict": num_predict, "seed": seed},
    )
    if fmt is not None:
        kwargs["format"] = fmt
    client = _get_client()
    try:
        try:
            response = client.chat(think=think, **kwargs)
        except TypeError:                      # older ollama library without `think`
            response = client.chat(**kwargs)
    except Exception as e:
        if ACTIVE_MODE == "online" and _gemini_key() and _is_payment_error(e):
            _ollama_blocked_until = time.time() + 600      # Ollama Cloud needs credits: use Gemini for 10 min
            return _gemini_generate(prompt, fmt, temperature, num_predict, seed)
        raise

    msg = _field(response, "message")
    content = _field(msg, "content") or ""
    thinking = _field(msg, "thinking") or ""
    if not content.strip() and thinking.strip():
        content = thinking                 # answer is sometimes only inside the thinking text
    return content


def _clean(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return re.sub(r"```(?:json)?", "", text).strip()


def parse_score(text):
    """Return (score, feedback); either can be None if not found."""
    text = _clean(text)
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


# -------------------------------------------------------------------------
# Public functions
# -------------------------------------------------------------------------
def grade_answer(question, max_marks, rubric, reference, answer, max_attempts=6, progress=None):
    """
    Grade one answer. Returns {"score", "feedback", "attempt", "seconds"}.
    Raises GradingError if no valid score + feedback could be obtained.
    `progress` is an optional function called with a status string.
    """
    max_marks = float(max_marks)
    answer = " ".join(str(answer).split())          # collapse line breaks / extra spaces from OCR
    if not answer:
        raise GradingError("The student answer is empty.")

    started = time.time()
    last_error = "unknown error"
    for attempt in range(1, max_attempts + 1):
        think, fmt, temp, n_predict, short = ATTEMPTS[(attempt - 1) % len(ATTEMPTS)]
        builder = build_short_prompt if short else build_prompt
        prompt = builder(question, max_marks, rubric, reference, answer)
        if progress:
            progress(f"Grading... attempt {attempt}/{max_attempts}")
        try:
            raw = _call_model(prompt, think, fmt, temp, n_predict, seed=attempt * 7 + 1)
            score, feedback = parse_score(raw)
            if score is None:
                last_error = f"no score in model output: {raw[:120]!r}"
            elif not feedback:
                last_error = f"no feedback in model output: {raw[:120]!r}"
            else:
                return {
                    "score": round(max(0.0, min(score, max_marks)), 2),
                    "feedback": feedback,
                    "attempt": attempt,
                    "seconds": round(time.time() - started, 1),
                }
        except Exception as e:
            last_error = str(e)
        time.sleep(1)

    raise GradingError(f"Could not get a valid grade after {max_attempts} attempts. Last problem: {last_error}")


def suggest_rubric(question, max_marks, subject=""):
    """
    Ask the model for key points + a short reference answer for a NEW question.
    Returns {"rubric": str, "reference_answer": str}. The teacher must review both.
    """
    topic = f" in {subject}" if subject else ""
    prompt = f"""You are an experienced university examiner{topic}.

For the exam question below, write the 4 most important key points a correct answer must cover,
and a concise model reference answer (3-5 sentences) that contains those points.
Be factually correct. Each key point must be one short sentence.

QUESTION:
{question}

Respond ONLY with JSON in this form:
{{"key_points": ["point 1", "point 2", "point 3", "point 4"], "reference_answer": "model answer"}}"""

    configs = [(False, RUBRIC_SCHEMA, 0.0, 1500), (False, "json", 0.3, 1500), (True, None, 0.3, 6000)]
    last_error = "unknown error"
    for i, (think, fmt, temp, n_predict) in enumerate(configs, 1):
        try:
            raw = _clean(_call_model(prompt, think, fmt, temp, n_predict, seed=i * 11))
            for block in reversed(re.findall(r"\{[^{}]*\}", raw, re.DOTALL)):
                try:
                    data = json.loads(block)
                except Exception:
                    continue
                points = [str(p).strip() for p in data.get("key_points", []) if str(p).strip()]
                reference = str(data.get("reference_answer", "")).strip()
                if 2 <= len(points) <= 6 and reference:
                    return {"rubric": _rubric_from_points(points, max_marks), "reference_answer": reference}
            last_error = f"unusable output: {raw[:120]!r}"
        except Exception as e:
            last_error = str(e)
    raise GradingError(f"Could not generate a rubric. Last problem: {last_error}")


def clean_ocr_text(text):
    """
    Optional: ask the model to fix obvious OCR spelling mistakes WITHOUT changing the meaning
    or adding anything. Returns the cleaned text. The teacher should still review it.
    """
    text = (text or "").strip()
    if not text:
        raise GradingError("There is no text to clean.")
    prompt = f"""The text below was read from a student's handwriting by OCR and contains recognition mistakes.
Fix ONLY obvious spelling and word-recognition errors. Do NOT add, remove or improve any content,
do NOT correct the student's facts, and keep the original line breaks.

TEXT:
{text}

Respond ONLY with JSON in this form:
{{"cleaned": "<corrected text>"}}"""
    last_error = "unknown error"
    for i, (think, fmt, temp, n_predict) in enumerate(
            [(False, {"type": "object", "properties": {"cleaned": {"type": "string"}}, "required": ["cleaned"]}, 0.0, 2048),
             (False, "json", 0.2, 2048)], 1):
        try:
            raw = _clean(_call_model(prompt, think, fmt, temp, n_predict, seed=i * 13))
            for block in reversed(re.findall(r"\{.*\}", raw, re.DOTALL)):
                try:
                    cleaned = str(json.loads(block).get("cleaned", "")).strip()
                except Exception:
                    continue
                if cleaned:
                    return cleaned
            last_error = f"unusable output: {raw[:120]!r}"
        except Exception as e:
            last_error = str(e)
    raise GradingError(f"Could not clean the text. Last problem: {last_error}")
