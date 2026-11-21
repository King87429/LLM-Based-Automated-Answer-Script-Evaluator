"""
AI Grading Dashboard - teacher-facing app for rubric-based grading of handwritten answers.

Run:   streamlit run app.py
Local mode : Ollama running with  deepseek-r1:8b  installed  (ollama pull deepseek-r1:8b)
Online mode: OLLAMA_API_KEY in Streamlit Secrets (Ollama Cloud)
"""
import difflib
import hashlib
import re
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

import grader
import ocr
import pdf_report

APP_DIR = Path(__file__).resolve().parent
REPORTS_DIR = APP_DIR / "reports"

# Leave empty to offer every subject found in the dataset, or list the ones you want, e.g.
# ALLOWED_SUBJECTS = ["Machine Learning", "Deep Learning", "NLP", "DBMS", "Operating Systems"]
ALLOWED_SUBJECTS = []

FALLBACK_SUBJECTS = [
    "Machine Learning", "Deep Learning", "NLP", "DBMS", "Operating Systems", "Computer Networks",
    "Data Structures & Algorithms", "Artificial Intelligence", "Computer Architecture",
    "Software Engineering", "Cybersecurity", "Cloud Computing",
]
OWN_QUESTION = "- write my own question -"
MODE_LOCAL = "Local (DeepSeek-R1:8B)"
MODE_ONLINE = "Online (Ollama Cloud)"

st.set_page_config(page_title="AI Grading Dashboard", page_icon="📝", layout="wide")


# ------------------------------------------------------------------ helpers
# ------------------------------------------------------------------ copied-answer check
# Every graded answer is saved in reports/submissions_log.csv. A new answer to the same question is
# compared with the earlier ones (fuzzy matching, so OCR differences do not hide a copy).
LOG_PATH = REPORTS_DIR / "submissions_log.csv"
LOG_COLUMNS = ["entry_id", "timestamp", "student", "subject", "question", "answer",
               "image_hash", "score", "max_marks"]
MIN_WORDS = 8             # answers shorter than this are not compared (too easy to match by chance)
SIMILARITY_WARN = 0.75    # show a warning at or above this similarity
SIMILARITY_HIGH = 0.90    # show a strong warning at or above this similarity


def load_log():
    if LOG_PATH.exists():
        try:
            df = pd.read_csv(LOG_PATH, dtype=str, keep_default_na=False)
            for c in LOG_COLUMNS:
                if c not in df.columns:
                    df[c] = ""
            return df[LOG_COLUMNS]
        except Exception:
            pass
    return pd.DataFrame(columns=LOG_COLUMNS)


def save_log(df):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(LOG_PATH, index=False)


def clear_log():
    LOG_PATH.unlink(missing_ok=True)
    st.session_state["confirm_clear"] = False


def _norm(text):
    """Lower-case and keep only letters/digits, so OCR symbols like _ ~ | do not matter."""
    return " ".join(re.sub(r"[\W_]+", " ", str(text).lower()).split())


def text_similarity(a, b):
    """0..1 similarity of two normalised answers. Uses character matching AND 3-word phrase overlap."""
    wa, wb = a.split(), b.split()
    if len(wa) < MIN_WORDS or len(wb) < MIN_WORDS:
        return 0.0
    seq = difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()
    sa, sb = set(zip(wa, wa[1:], wa[2:])), set(zip(wb, wb[1:], wb[2:]))
    overlap = len(sa & sb) / len(sa | sb) if sa and sb else 0.0     # still works if lines were re-ordered
    return max(seq, overlap)


def find_similar(log, question, answer, student, img_hash):
    """Return (matches, compared). `compared` = earlier answers to the same question that were checked."""
    qn, an, who = _norm(question), _norm(answer), student.strip().lower()
    matches, compared = [], 0
    for row in log.itertuples():
        row_q = _norm(row.question)
        same_q = row_q == qn or difflib.SequenceMatcher(None, qn, row_q).ratio() >= 0.85
        same_photo = bool(img_hash) and row.image_hash == img_hash
        if not (same_q or same_photo):
            continue
        if who and row.student.strip().lower() == who:
            continue                                   # same student graded again - not a copy
        if same_q:
            compared += 1
        sim = text_similarity(an, _norm(row.answer)) if same_q else 0.0
        if sim >= SIMILARITY_WARN or same_photo:
            matches.append({"student": row.student or "(no name)", "when": row.timestamp, "similarity": sim,
                            "same_photo": same_photo, "score": row.score, "max_marks": row.max_marks})
    matches.sort(key=lambda m: (m["same_photo"], m["similarity"]), reverse=True)
    return matches[:5], compared


def similarity_lines(matches, compared):
    """Text lines for the screen and the PDF, plus a flag saying whether something was found."""
    if not matches:
        if compared:
            return [f"No similar answer found (compared with {compared} earlier answer(s) to this question)."], False
        return ["No earlier answers to this question were available for comparison."], False
    lines = []
    for m in matches:
        if m["same_photo"]:
            lines.append(f"Exactly the same photo was already graded for {m['student']} on {m['when']}.")
        else:
            lines.append(f"{m['similarity']:.0%} similar to the answer of {m['student']} "
                         f"(graded {m['when']}, {m['score']}/{m['max_marks']} marks).")
    return lines, True


@st.cache_data(show_spinner=False)
def load_question_bank():
    """Optional: sample questions from the research dataset, if the CSV is next to / above this folder."""
    for folder in (APP_DIR, APP_DIR.parent):
        for name in ("dataset_500.csv", "dataset.csv"):
            path = folder / name
            if not path.exists():
                continue
            try:
                df = pd.read_csv(path)
                need = ["subject", "question", "max_score", "rubric", "reference_answer"]
                if all(c in df.columns for c in need):
                    df = df[need].dropna(subset=["question"]).drop_duplicates(subset=["question"])
                    return df.reset_index(drop=True)
            except Exception:
                continue
    return None


@st.cache_data(ttl=15, show_spinner=False)
def ollama_status():
    return grader.check_ollama()


@st.cache_data(ttl=15, show_spinner=False)
def engine_status(mode):          # `mode` only makes the cache separate for Local / Online
    return ocr.available_engines()


def load_sample(subject, bank):
    choice = st.session_state.get(f"sample_{subject}")
    row = bank[bank["question"] == choice]
    if choice and not row.empty:
        r = row.iloc[0]
        ss = st.session_state
        ss["question"] = str(r["question"])
        ss["max_marks"] = float(r["max_score"])
        ss["rubric"] = str(r["rubric"])
        ss["reference"] = str(r["reference_answer"])
        ss["rubric_auto"] = None


def reset_all():
    ss = st.session_state
    for key in ("question", "rubric", "reference", "answer_text", "student"):
        ss[key] = ""
    ss["max_marks"] = 10.0
    ss["rubric_auto"] = None
    ss["ocr_conf"] = None
    ss["ocr_engine"] = ""
    ss["ocr_raw"] = ""
    ss["ocr_note"] = ""
    ss["result"] = None
    ss["uploader_n"] += 1


# ------------------------------------------------------------------ state
ss = st.session_state
ss.setdefault("max_marks", 10.0)
for k in ("question", "rubric", "reference", "answer_text", "student"):
    ss.setdefault(k, "")
ss.setdefault("rubric_auto", None)
ss.setdefault("ocr_conf", None)
ss.setdefault("ocr_engine", "")
ss.setdefault("ocr_raw", "")
ss.setdefault("ocr_note", "")
ss.setdefault("result", None)
ss.setdefault("uploader_n", 0)

# values produced by slow actions (OCR / AI rubric) are applied here, BEFORE the widgets are drawn
pending = ss.pop("_pending", None)
if pending:
    for k, v in pending.items():
        ss[k] = v

# keep the general rubric template in step with the maximum marks until the teacher edits it
if not ss["rubric"].strip() or ss["rubric"] == ss["rubric_auto"]:
    ss["rubric"] = grader.default_rubric(ss["max_marks"])
    ss["rubric_auto"] = ss["rubric"]

bank = load_question_bank()
subjects = sorted(bank["subject"].unique()) if bank is not None else FALLBACK_SUBJECTS
if ALLOWED_SUBJECTS:
    subjects = [s for s in subjects if s in ALLOWED_SUBJECTS] or subjects

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("📝 AI Grading Dashboard")
    if "mode_label" not in ss:      # first load: Local if Ollama is reachable on this machine, else Online
        ss["mode_label"] = MODE_LOCAL if ollama_status()[0] else MODE_ONLINE
    st.radio("Grading mode", [MODE_LOCAL, MODE_ONLINE], key="mode_label")
    grader.set_mode("local" if ss["mode_label"] == MODE_LOCAL else "online")   # BEFORE any grading call
    if ss["mode_label"] == MODE_LOCAL:
        ok, msg = ollama_status()
        (st.success if ok else st.error)(msg)
    else:
        ok, msg = grader.check_cloud()
        (st.success if ok else st.error)(msg)
    st.markdown(
        "**How it works**\n"
        "1. Enter the question, marks and rubric\n"
        "2. Upload a photo of the handwritten answer\n"
        "3. Extract the text and **correct any mistakes**\n"
        "4. Grade, then download the PDF report"
    )
    st.caption("Marks and feedback come from an AI model "
               f"({'DeepSeek-R1:8B, local' if ss['mode_label'] == MODE_LOCAL else 'Ollama Cloud'}). "
               "They support the teacher and do not replace a human evaluator.")
    st.button("Start a new answer", on_click=reset_all)
    with st.expander("Class answer log (copy check)"):
        log_df = load_log()
        st.caption(f"{len(log_df)} graded answer(s) saved. New answers are compared with these.")
        if len(log_df):
            st.dataframe(log_df[["timestamp", "student", "subject", "score", "max_marks"]])
            st.download_button("Download log (CSV)", log_df.to_csv(index=False), "submissions_log.csv", "text/csv")
        st.checkbox("Yes, I want to clear the whole log", key="confirm_clear")
        st.button("Clear log", on_click=clear_log, disabled=not ss.get("confirm_clear", False))

st.title("Handwritten Answer Grading")

# ------------------------------------------------------------------ step 1 and 2
left, right = st.columns(2, gap="large")

with left:
    st.subheader("1. Question and rubric")
    subject = st.selectbox("Subject / topic", subjects, key="subject")

    if bank is not None:
        options = [OWN_QUESTION] + bank[bank["subject"] == subject]["question"].tolist()
        st.selectbox("Sample question from the dataset (optional)", options, key=f"sample_{subject}")
        if st.session_state.get(f"sample_{subject}", OWN_QUESTION) != OWN_QUESTION:
            st.button("Use this sample question", on_click=load_sample, args=(subject, bank))

    st.text_area("Question", key="question", height=90,
                 placeholder="Type any question from the selected subject")
    st.number_input("Maximum marks", min_value=1.0, max_value=100.0, step=0.5, key="max_marks")
    st.text_area("Rubric (edit so it matches this question)", key="rubric", height=170)
    st.text_area("Reference answer (optional)", key="reference", height=100)
    st.caption("The prefilled rubric is a general template. For best results, click the button below "
               "to draft a question-specific rubric, then review it, or write your own.")

    if st.button("✨ Suggest rubric and reference answer with AI"):
        if not ss["question"].strip():
            st.warning("Enter the question first.")
        else:
            try:
                with st.spinner("Drafting rubric... (can take a minute)"):
                    out = grader.suggest_rubric(ss["question"], ss["max_marks"], subject)
                ss["_pending"] = {"rubric": out["rubric"], "reference": out["reference_answer"],
                                  "rubric_auto": None}
                st.rerun()
            except grader.GradingError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Model error: {e}")

with right:
    st.subheader("2. Handwritten answer")
    uploaded = st.file_uploader("Upload a photo (JPG, JPEG or PNG only)", type=["jpg", "jpeg", "png"],
                                key=f"uploader_{ss['uploader_n']}")
    image_bytes = uploaded.getvalue() if uploaded else None

    engines = engine_status(ss["mode_label"])
    if ss["mode_label"] == MODE_ONLINE and not engines["vision_model"]:
        st.info("Online mode: no cloud vision model could be used, so EasyOCR is used. "
                + (f"Reason: {ocr.VISION_PROBLEM} " if ocr.VISION_PROBLEM else "")
                + "Handwriting accuracy is limited - please correct the extracted text before grading.")
    elif not ollama_status()[0] and not engines["vision_model"]:
        st.warning("Ollama is not running, so the vision model cannot be used. Start Ollama first "
                   "(see the red message in the sidebar). EasyOCR can still be used meanwhile.")
    elif engines["vision_model"]:
        st.caption(f"Handwriting reader: vision model **{engines['vision_model']}** (best accuracy)")
    else:
        st.warning("No vision model installed - EasyOCR will be used, which is weak on handwriting. "
                   "For much better results run once in a terminal:  `ollama pull qwen2.5vl:7b`")
    engine_label = st.radio("OCR engine", ["Auto (best available)", "Vision model", "EasyOCR"],
                            horizontal=True, key="ocr_engine_choice")
    engine = {"Auto (best available)": "auto", "Vision model": "vision", "EasyOCR": "easyocr"}[engine_label]

    if image_bytes:
        st.image(image_bytes, caption=uploaded.name)
        if st.button("🔍 Extract text from image"):
            try:
                with st.spinner("Reading handwriting... (a vision model can take 1-2 minutes)"):
                    out = ocr.read_image(image_bytes, engine)
                ss["_pending"] = {"answer_text": out["text"], "ocr_conf": out["confidence"],
                                  "ocr_engine": out["engine"], "ocr_raw": out["text"],
                                  "ocr_note": out.get("note", "")}
                st.rerun()
            except Exception as e:
                st.error(f"Could not read the image. {e}")
                st.info("You can still type or paste the answer in the box below.")

    st.text_area("Extracted answer - review and correct before grading", key="answer_text", height=220,
                 placeholder="The text read from the image appears here. Fix any OCR mistakes.")
    st.caption("If the photo also contains the question or marks, delete those lines from the box above - "
               "only the student's answer should be graded.")
    if ss.get("ocr_note"):
        st.warning("The vision model could not be used, so EasyOCR was used instead (much weaker on "
                   f"handwriting). Reason: {ss['ocr_note']}")
    if ss.get("ocr_engine"):
        msg = f"Text read using: {ss['ocr_engine']}."
        if ss["ocr_conf"] is None:
            st.caption(msg + " Handwriting reading can make mistakes - check every line.")
        elif ss["ocr_conf"] < 0.6:
            st.warning(msg + f" Confidence is low (about {ss['ocr_conf']:.0%}). Check every line carefully.")
        else:
            st.caption(msg + f" Confidence about {ss['ocr_conf']:.0%}. Please still review the text.")
    if ss["answer_text"].strip():
        if st.button("✨ Clean OCR mistakes with AI (optional)"):
            try:
                with st.spinner("Fixing OCR mistakes... (can take a minute)"):
                    cleaned = grader.clean_ocr_text(ss["answer_text"])
                pend = {"answer_text": cleaned}
                if not ss["ocr_raw"]:
                    pend["ocr_raw"] = ss["answer_text"]
                ss["_pending"] = pend
                st.rerun()
            except grader.GradingError as e:
                st.warning(str(e))
            except Exception as e:
                st.error(f"Model error: {e}")
    if ss["ocr_raw"] and ss["ocr_raw"] != ss["answer_text"]:
        with st.expander("Show the original OCR text (before cleaning / editing)"):
            st.text(ss["ocr_raw"])
    st.text_input("Student name / roll no. (optional)", key="student")

# ------------------------------------------------------------------ step 3
st.divider()
st.subheader("3. Grade")
include_image = st.checkbox("Include the handwritten image in the PDF report", value=True)

if st.button("Grade answer", type="primary"):
    problems = []
    if not ss["question"].strip():
        problems.append("Enter the question.")
    if not ss["rubric"].strip():
        problems.append("Enter the rubric.")
    if not ss["answer_text"].strip():
        problems.append("The student answer is empty. Extract the text from the image or type it.")
    if problems:
        for p in problems:
            st.warning(p)
    else:
        try:
            # copied-answer check (runs before grading; the log is only updated after grading succeeds)
            img_hash = hashlib.sha256(image_bytes).hexdigest() if image_bytes else ""
            # a NEW upload (even of the same file) is a new submission; only a repeated click on
            # "Grade answer" with the same upload counts as grading the same answer again
            upload_id = str(getattr(uploaded, "file_id", "")) if uploaded else ""
            sig = hashlib.sha1("|".join([_norm(ss["question"]), ss["student"].strip().lower(),
                                         _norm(ss["answer_text"]), img_hash, upload_id,
                                         str(ss["max_marks"])]).encode()).hexdigest()
            prev = ss["result"]
            regrade = bool(prev) and prev.get("sig") == sig          # same inputs graded again = not a copy
            entry_id = prev["entry_id"] if regrade else uuid.uuid4().hex
            log = load_log()
            if regrade:
                log = log[log["entry_id"] != entry_id]
            matches, compared = find_similar(log, ss["question"], ss["answer_text"], ss["student"], img_hash)
            sim_lines, sim_flag = similarity_lines(matches, compared)

            box = st.empty()
            with st.spinner(f"Grading with {grader.get_model_name()}... this can take 1-3 minutes"):
                res = grader.grade_answer(ss["question"], ss["max_marks"], ss["rubric"], ss["reference"],
                                          ss["answer_text"], progress=lambda m: box.caption(m))
            box.empty()
            data = {
                "student": ss["student"], "subject": subject, "question": ss["question"],
                "max_marks": ss["max_marks"], "rubric": ss["rubric"], "reference": ss["reference"],
                "answer": ss["answer_text"], "score": res["score"], "feedback": res["feedback"],
                "model": grader.get_model_name(), "timestamp": datetime.now(),
                "similarity_lines": sim_lines, "similarity_flag": sim_flag,
            }
            pdf_bytes = pdf_report.build_report(data, image_bytes if include_image else None)
            saved = pdf_report.save_report(pdf_bytes, REPORTS_DIR)

            new_row = {"entry_id": entry_id, "timestamp": data["timestamp"].strftime("%Y-%m-%d %H:%M"),
                       "student": ss["student"].strip(), "subject": subject, "question": ss["question"],
                       "answer": ss["answer_text"], "image_hash": img_hash,
                       "score": f"{res['score']:g}", "max_marks": f"{float(ss['max_marks']):g}"}
            try:
                save_log(pd.concat([log, pd.DataFrame([new_row])], ignore_index=True))
            except Exception as e:
                st.warning(f"The result could not be saved to the answer log ({e}). "
                           "If the log file is open in Excel, close it.")
            ss["result"] = {**data, "seconds": res["seconds"], "pdf": pdf_bytes, "pdf_path": saved,
                            "sig": sig, "entry_id": entry_id, "similar": matches, "compared": compared}
        except grader.GradingError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Something went wrong: {e}")

# ------------------------------------------------------------------ results
r = ss["result"]
if r:
    st.success("Grading complete")
    c1, c2, c3 = st.columns(3)
    c1.metric("Marks awarded", f"{r['score']:g} / {r['max_marks']:g}")
    c2.metric("Percentage", f"{r['score'] / r['max_marks'] * 100:.1f}%")
    c3.metric("Time taken", f"{r['seconds']} s")
    st.progress(min(1.0, max(0.0, r["score"] / r["max_marks"])))
    st.markdown("**Feedback**")
    st.info(r["feedback"])

    st.markdown("**Copied-answer check**")
    if r["similar"]:
        top = r["similar"][0]
        strong = top["same_photo"] or top["similarity"] >= SIMILARITY_HIGH
        (st.error if strong else st.warning)(
            "⚠️ " + ("Very similar to an earlier answer." if strong else "Similar to an earlier answer.")
            + " Please compare both answers before using these marks.")
        st.markdown("\n".join(f"- {line}" for line in r["similarity_lines"]))
        st.caption("Similar answers can happen by chance (short or textbook-style answers). "
                   "This is a prompt for the teacher to check, not proof of copying.")
    elif r["compared"]:
        st.success(f"✅ {r['similarity_lines'][0]}")
    else:
        st.caption(r["similarity_lines"][0])
    st.download_button("⬇️ Download PDF report", data=r["pdf"],
                       file_name=f"grading_report_{r['timestamp'].strftime('%Y%m%d_%H%M%S')}.pdf",
                       mime="application/pdf")
    st.caption(f"A copy was also saved to: {r['pdf_path']}")
    st.caption("AI-assisted result. Please review before using the marks.")
