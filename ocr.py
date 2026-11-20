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
def _ollama_client():
    from ollama import Client
    return Client(timeout=VISION_TIMEOUT)


def installed_vision_model():
    """Return the name of the best installed vision model, or None."""
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


def _read_with_vision(image_bytes):
    model = installed_vision_model()
    if not model:
        raise RuntimeError("No vision model is installed in Ollama. Run:  ollama pull qwen2.5vl:7b")
    img = _resize(_open_rgb(image_bytes), 1600)          # keep colour, moderate size
    response = _ollama_client().chat(
        model=model,
        messages=[{"role": "user", "content": VISION_PROMPT, "images": [_to_jpeg_bytes(img)]}],
        options={"temperature": 0, "num_ctx": 8192, "num_predict": 2048},
    )
    msg = response["message"] if isinstance(response, dict) else response.message
    content = msg["content"] if isinstance(msg, dict) else msg.content
    return _clean_transcription(content), None, f"Vision model ({model})"


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
    order = {"auto": [_read_with_vision, _read_with_easyocr],
             "vision": [_read_with_vision],
             "easyocr": [_read_with_easyocr]}[engine]

    problems = []
    for fn in order:
        name = "Vision model" if fn is _read_with_vision else "EasyOCR"
        try:
            text, conf, used = fn(image_bytes)
            if text.strip():
                return {"text": text, "confidence": conf, "engine": used}
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