"""
ocr.py - reads a photo of a handwritten answer (JPG / JPEG / PNG) and returns text.

Two engines are supported:
  1. VISION MODEL through Ollama (recommended for handwriting - much more accurate).
     Install once with:   ollama pull qwen2.5vl:7b        (lighter option: ollama pull gemma3:4b)
  2. EasyOCR (works without a vision model, but is weak on handwriting).

engine="auto" tries the vision model first and falls back to EasyOCR.
Handwriting OCR is never perfect, so the dashboard always lets the teacher correct the text.

Test from the terminal (no Streamlit needed):
    python ocr.py path/to/photo.jpg
"""
import base64
import io
import re
import statistics
import sys

import numpy as np
from PIL import Image, ImageOps

# vision models to look for, best first. Put your own at the top to force one.
VISION_MODELS = ["qwen2.5vl:7b", "qwen2.5vl", "qwen2.5vl:3b", "minicpm-v", "llama3.2-vision",
                 "gemma3:12b", "gemma3:4b", "llava"]
VISION_TIMEOUT = 300          # seconds

VISION_PROMPT = (
    "This is a photo of a handwritten exam answer on ruled notebook paper. "
    "Transcribe ALL the handwritten text exactly as written, from top to bottom. "
    "Ignore the ruled lines, margins and underlines. Keep the line breaks and bullet points. "
    "Do not correct spelling, do not summarise, do not explain. "
    "If a word is unreadable write [?]. Output ONLY the transcription."
)

_reader = None


# ------------------------------------------------------------------ image preparation
def _open_rgb(image_bytes):
    """Open any JPG/PNG, fix phone rotation, and flatten transparency onto white."""
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes)))
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        white = Image.new("RGBA", img.size, "white")
        img = Image.alpha_composite(white, img)
    return img.convert("RGB")


def _resize(img, longest_target):
    longest = max(img.size)
    if longest > longest_target:
        s = longest_target / longest
        return img.resize((int(img.width * s), int(img.height * s)), Image.LANCZOS)
    return img


def _to_jpeg_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


# ------------------------------------------------------------------ engine 1: vision model
# Vision models offered by Ollama Cloud, best first. Names change over time (qwen3-vl cloud tags were
# retired), so if none of these exist the code asks the catalogue which models have "vision".
# Add  OLLAMA_CLOUD_VISION_MODEL = "..."  to Streamlit Secrets to force a specific one.
CLOUD_VISION_MODELS = [
    "qwen3.5:397b", "qwen3.5", "gemma4:31b", "gemma4", "qwen3.6:35b", "qwen3.6", "qwen3.8:27b", "qwen3.8",
    "ministral-3:14b", "ministral-3:8b", "ministral-3:3b", "gemma3:27b", "gemma3:12b", "gemma3:4b", "gemini-3-flash-preview",
    "qwen3-vl:235b-instruct", "qwen3-vl:235b",
]
_cloud_vision_cache = None          # ordered list of usable cloud vision models
VISION_PROBLEM = ""                 # human-readable reason when no cloud vision model could be used


def _online():
    """True when the dashboard is in Online (Ollama Cloud) mode."""
    try:
        import grader
        return grader.ACTIVE_MODE == "online"
    except Exception:
        return False


def _has_vision(client, name):
    try:
        info = client.show(name)
        caps = info["capabilities"] if isinstance(info, dict) else getattr(info, "capabilities", None)
        return bool(caps) and "vision" in [str(c).lower() for c in caps]
    except Exception:
        return False


def _cloud_vision_models():
    """Ordered list of vision-capable models in the Ollama Cloud catalogue (empty if none / no key)."""
    global _cloud_vision_cache, VISION_PROBLEM
    if _cloud_vision_cache:
        return _cloud_vision_cache
    import grader
    if not grader.get_api_key():
        VISION_PROBLEM = "OLLAMA_API_KEY is missing."
        return []
    forced = grader._secret("OLLAMA_CLOUD_VISION_MODEL").strip()
    try:
        client = grader._cloud_client(timeout=20)
        resp = client.list()
        models = resp["models"] if isinstance(resp, dict) else getattr(resp, "models", [])
        names = []
        for m in models:
            n = m.get("model") if isinstance(m, dict) else getattr(m, "model", None)
            if n:
                names.append(str(n))
    except Exception as e:
        VISION_PROBLEM = f"Could not read the Ollama Cloud model list ({str(e)[:80]})."
        return [forced] if forced else []

    picks = [c for c in ([forced] if forced else []) + CLOUD_VISION_MODELS if c in names]
    if len(picks) < 6:                             # also ask the catalogue which models support vision
        for n in names[:40]:
            if n not in picks and _has_vision(client, n):
                picks.append(n)
            if len(picks) >= 7:
                break
    if picks:
        _cloud_vision_cache = picks
        VISION_PROBLEM = ""
    else:
        VISION_PROBLEM = f"No vision model found among {len(names)} cloud models."
    return picks


def _cloud_vision_model():
    picks = _cloud_vision_models()
    return picks[0] if picks else None


def _ollama_client():
    from ollama import Client
    if _online():
        import grader
        return grader._cloud_client(timeout=VISION_TIMEOUT)
    return Client(timeout=VISION_TIMEOUT)


def installed_vision_model():
    """Return the name of the best available vision model (local install, Gemini or Ollama Cloud), or None."""
    if _online():
        if _gemini_key():
            return f"{_gemini_models()[0]} (Google Gemini)"
        return _cloud_vision_model()
    try:
        from ollama import Client
        resp = Client(timeout=5).list()
        models = resp["models"] if isinstance(resp, dict) else getattr(resp, "models", [])
        names = []
        for m in models:
            n = m.get("model") if isinstance(m, dict) else getattr(m, "model", None)
            if n:
                names.append(str(n))
    except Exception:
        return None
    for wanted in VISION_MODELS:
        for n in names:
            if n == wanted or (":" not in wanted and n.split(":")[0] == wanted):
                return n
    return None


def _clean_transcription(text):
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"```[a-zA-Z]*", "", text).strip()
    lines = text.splitlines()
    # drop a chatty first line such as "Here is the transcription:"
    if lines and re.match(r"^(here|the (handwritten )?text|transcription|sure)\b.*:\s*$", lines[0].strip(), re.I):
        lines = lines[1:]
    return "\n".join(l.rstrip() for l in lines).strip()


def _chat_vision(client, model, jpeg, num_predict):
    kwargs = dict(model=model,
                  messages=[{"role": "user", "content": VISION_PROMPT, "images": [jpeg]}],
                  options={"temperature": 0, "num_ctx": 8192, "num_predict": num_predict})
    try:
        response = client.chat(think=False, **kwargs)     # thinking models would waste the token budget
    except Exception:
        response = client.chat(**kwargs)                  # old client / model without a thinking switch
    msg = response["message"] if isinstance(response, dict) else response.message
    content = msg["content"] if isinstance(msg, dict) else msg.content
    return _clean_transcription(content)


def _read_with_vision(image_bytes):
    online = _online()
    models = _cloud_vision_models()[:7] if online else [installed_vision_model()]
    models = [m for m in models if m]
    if not models:
        raise RuntimeError("No vision model found in Ollama Cloud. " + (VISION_PROBLEM or "Check OLLAMA_API_KEY.")
                           if online else "No vision model is installed in Ollama. Run:  ollama pull qwen2.5vl:7b")
    img = _resize(_open_rgb(image_bytes), 1600)          # keep colour, moderate size
    jpeg = _to_jpeg_bytes(img)
    client = _ollama_client()
    errors = []
    for model in models:                                 # cloud: try the next model if one fails
        try:
            text = _chat_vision(client, model, jpeg, 4096 if online else 2048)
            if text.strip():
                return text, None, f"Vision model ({model})"
            errors.append(f"{model}: empty reply")
        except Exception as e:
            errors.append(f"{model}: {str(e)[:120]}")
    raise RuntimeError("; ".join(errors))


# ------------------------------------------------------------------ engine 1b: Google Gemini (optional, online)
# Free API key from https://aistudio.google.com/apikey -> add to Streamlit Secrets:  GEMINI_API_KEY = "..."
# Optional:  GEMINI_MODEL = "gemini-2.5-flash"  to force a model.
GEMINI_MODELS = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.5-flash-lite"]


def _gemini_key():
    try:
        import grader
        return grader._secret("GEMINI_API_KEY").strip()
    except Exception:
        return ""


def _gemini_models():
    import grader
    forced = grader._secret("GEMINI_MODEL").strip()
    return [forced] + [m for m in GEMINI_MODELS if m != forced] if forced else list(GEMINI_MODELS)


def _read_with_gemini(image_bytes):
    import requests
    key = _gemini_key()
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    jpeg = _to_jpeg_bytes(_resize(_open_rgb(image_bytes), 1600))
    payload = {"contents": [{"parts": [{"text": VISION_PROMPT},
                                       {"inline_data": {"mime_type": "image/jpeg",
                                                        "data": base64.b64encode(jpeg).decode()}}]}],
               "generationConfig": {"temperature": 0}}
    errors = []
    for model in _gemini_models():
        try:
            r = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                              headers={"x-goog-api-key": key}, json=payload, timeout=120)
            if r.status_code != 200:
                errors.append(f"{model}: HTTP {r.status_code} {r.text[:100]}")
                continue
            parts = r.json()["candidates"][0]["content"]["parts"]
            text = _clean_transcription("".join(p.get("text", "") for p in parts))
            if text.strip():
                return text, None, f"Gemini ({model})"
            errors.append(f"{model}: empty reply")
        except Exception as e:
            errors.append(f"{model}: {str(e)[:100]}")
    raise RuntimeError("; ".join(errors))


# ------------------------------------------------------------------ engine 2: EasyOCR
def get_reader():
    """Load EasyOCR once. The first run downloads the English model (needs internet)."""
    global _reader
    if _reader is None:
        try:
            import easyocr
        except ImportError as e:
            raise RuntimeError("EasyOCR is not installed. Run: pip install easyocr") from e
        try:
            import torch
            use_gpu = torch.cuda.is_available()
        except Exception:
            use_gpu = False
        try:
            _reader = easyocr.Reader(["en"], gpu=use_gpu)
        except Exception as e:
            raise RuntimeError(f"EasyOCR could not load its model (needs internet on the first run): {e}") from e
    return _reader


def _estimate_skew(img, max_angle=8.0):
    """Estimate how many degrees the page is tilted (works well on ruled / lined paper)."""
    from PIL import ImageFilter
    g = ImageOps.grayscale(img)
    w = 700
    g = g.resize((w, max(1, int(g.height * w / g.width))), Image.BILINEAR)
    a = np.asarray(g, dtype=np.float32)
    background = np.asarray(g.filter(ImageFilter.BoxBlur(25)), dtype=np.float32)
    ink = ((np.clip(background - a, 0, None)) > 12).astype(np.uint8) * 255
    ink_img = Image.fromarray(ink)

    def score(angle):
        r = np.asarray(ink_img.rotate(angle, resample=Image.BILINEAR, fillcolor=0), dtype=np.float32)
        return float(np.sum(np.diff(r.sum(axis=1)) ** 2))   # sharp row profile = straight lines

    coarse = max(np.arange(-max_angle, max_angle + 0.01, 1.0), key=score)
    return float(max(np.arange(coarse - 1, coarse + 1.01, 0.2), key=score))


def _deskew(img):
    try:
        angle = _estimate_skew(img)
    except Exception:
        return img
    if abs(angle) < 0.5:
        return img
    return img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255))


def _easyocr_prepare(image_bytes):
    img = _deskew(_open_rgb(image_bytes))          # straighten tilted phone photos first
    if max(img.size) < 1200:
        img = img.resize((img.width * 2, img.height * 2), Image.LANCZOS)
    return _resize(img, 2400)


def _group_into_lines(results):
    """Turn EasyOCR boxes into text lines in reading order (top->bottom, left->right)."""
    items = []
    for bbox, text, conf in results:
        text = str(text).strip()
        if not text:
            continue
        xs = [float(p[0]) for p in bbox]
        ys = [float(p[1]) for p in bbox]
        items.append({"x": min(xs), "yc": sum(ys) / len(ys), "h": max(ys) - min(ys),
                      "text": text, "conf": float(conf)})
    if not items:
        return [], 0.0

    items.sort(key=lambda d: d["yc"])
    tolerance = 0.6 * statistics.median(d["h"] for d in items)
    lines = []
    for it in items:
        if lines and abs(it["yc"] - lines[-1]["yc"]) <= tolerance:
            lines[-1]["items"].append(it)
            lines[-1]["yc"] = sum(i["yc"] for i in lines[-1]["items"]) / len(lines[-1]["items"])
        else:
            lines.append({"yc": it["yc"], "items": [it]})

    text_lines = [" ".join(i["text"] for i in sorted(l["items"], key=lambda d: d["x"])) for l in lines]
    total = sum(len(i["text"]) for i in items)
    return text_lines, sum(i["conf"] * len(i["text"]) for i in items) / total


def _read_with_easyocr(image_bytes):
    reader = get_reader()
    color = _easyocr_prepare(image_bytes)
    gray = ImageOps.autocontrast(ImageOps.grayscale(color), cutoff=1)

    # settings tuned for handwriting (looser text detection than the defaults)
    params = dict(detail=1, paragraph=False, text_threshold=0.5, low_text=0.3, link_threshold=0.4,
                  canvas_size=2560, mag_ratio=1.5, contrast_ths=0.1, adjust_contrast=0.7, width_ths=0.7)

    best_text, best_conf = "", 0.0
    for candidate in (color, gray):                      # try both, keep the one with more text
        lines, conf = _group_into_lines(reader.readtext(np.array(candidate), **params))
        text = "\n".join(lines)
        if len(text) > len(best_text):
            best_text, best_conf = text, conf
    return best_text, best_conf, "EasyOCR"


# ------------------------------------------------------------------ public API
def available_engines():
    """Short description of what can be used right now (for the dashboard sidebar).
    Never imports EasyOCR/PyTorch here - that import is slow and would freeze the page."""
    import importlib.util
    return {"vision_model": installed_vision_model(),
            "easyocr": importlib.util.find_spec("easyocr") is not None}


def read_image(image_bytes, engine="auto"):
    """
    engine: "auto" | "vision" | "easyocr"
    Returns {"text": str, "confidence": float | None, "engine": str}.
    Raises RuntimeError listing every reason if nothing could read the image.
    """
    vision_fns = [_read_with_vision]
    if _online() and _gemini_key():
        vision_fns = [_read_with_gemini, _read_with_vision]       # Gemini first, Ollama Cloud as backup
    order = {"auto": vision_fns + [_read_with_easyocr],
             "vision": vision_fns,
             "easyocr": [_read_with_easyocr]}[engine]
    names = {_read_with_gemini: "Gemini", _read_with_vision: "Vision model", _read_with_easyocr: "EasyOCR"}

    problems = []
    for fn in order:
        name = names[fn]
        try:
            text, conf, used = fn(image_bytes)
            if text.strip():
                result = {"text": text, "confidence": conf, "engine": used}
                if problems:                       # an earlier engine failed - tell the teacher why
                    result["note"] = " | ".join(problems)
                return result
            problems.append(f"{name}: no text was found in the image")
        except Exception as e:
            problems.append(f"{name}: {e}")
    raise RuntimeError(" | ".join(problems))


def extract_text(image_bytes, engine="auto"):
    """Backward compatible helper: returns (text, confidence)."""
    r = read_image(image_bytes, engine)
    return r["text"], r["confidence"]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python ocr.py path/to/photo.jpg")
    print("Engines available:", available_engines())
    with open(sys.argv[1], "rb") as f:
        data = f.read()
    try:
        out = read_image(data)
        print(f"\nEngine used: {out['engine']}\n----- TEXT -----\n{out['text']}\n----------------")
    except Exception as e:
        print("\nOCR FAILED:", e)
