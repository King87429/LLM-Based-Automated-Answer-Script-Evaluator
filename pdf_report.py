"""
pdf_report.py - builds the downloadable PDF analysis report with ReportLab.
"""
import io
import os
from datetime import datetime
from xml.sax.saxutils import escape

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image as RLImage, KeepTogether, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

NAVY = colors.HexColor("#1F3A5F")
LIGHT = colors.HexColor("#F2F5F9")
GREY = colors.HexColor("#666666")


def _txt(text):
    """Make text safe for ReportLab: standard PDF fonts only support Latin-1/cp1252 characters."""
    text = "" if text is None else str(text)
    text = text.encode("cp1252", "replace").decode("cp1252")
    return escape(text).replace("\n", "<br/>")


def _num(x):
    return f"{round(float(x), 2):g}"


def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("title", parent=base["Title"], fontSize=20, textColor=NAVY, spaceAfter=2),
        "sub": ParagraphStyle("sub", parent=base["Normal"], fontSize=9, textColor=GREY, alignment=1, spaceAfter=10),
        "h": ParagraphStyle("h", parent=base["Heading3"], fontSize=11, textColor=NAVY, spaceBefore=10, spaceAfter=3),
        "box": ParagraphStyle("box", parent=base["Normal"], fontSize=10, leading=14, backColor=LIGHT,
                              borderPadding=(6, 6, 6, 6), leftIndent=6, rightIndent=6, spaceAfter=8),
        "boxwarn": ParagraphStyle("boxwarn", parent=base["Normal"], fontSize=10, leading=14,
                                  backColor=colors.HexColor("#FDECEA"), borderPadding=(6, 6, 6, 6),
                                  leftIndent=6, rightIndent=6, spaceAfter=8),
        "cell": ParagraphStyle("cell", parent=base["Normal"], fontSize=10, leading=13),
        "head": ParagraphStyle("head", parent=base["Normal"], fontSize=9, textColor=colors.white),
    }


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(GREY)
    canvas.drawString(2 * cm, 1.2 * cm,
                      "AI-assisted marking (DeepSeek-R1:8B). Marks must be reviewed by the teacher.")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
    canvas.restoreState()


def _image_flowable(image_bytes, max_w=16 * cm, max_h=6.5 * cm):
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert("RGB")
    img.thumbnail((1400, 1400))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    buf.seek(0)
    w, h = img.size
    scale = min(max_w / w, max_h / h)
    return RLImage(buf, width=w * scale, height=h * scale)


def build_report(data, image_bytes=None):
    """
    data keys: student, subject, question, max_marks, rubric, reference, answer,
               score, feedback, model, timestamp (datetime, optional)
    Returns the PDF as bytes.
    """
    st = _styles()
    max_marks = float(data["max_marks"])
    score = float(data["score"])
    percent = (score / max_marks * 100) if max_marks else 0
    when = data.get("timestamp") or datetime.now()

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=1.8 * cm, bottomMargin=2 * cm, title="AI Grading Analysis Report")
    story = [
        Paragraph("AI Grading - Analysis Report", st["title"]),
        Paragraph(f"Generated on {when.strftime('%d %B %Y, %H:%M')}", st["sub"]),
    ]

    # ---- summary table
    def head(t): return Paragraph(t, st["head"])
    def cell(t): return Paragraph(_txt(t), st["cell"])
    rows = [
        [head("Student"), head("Subject"), head("Maximum marks"), head("Marks awarded"), head("Percentage")],
        [cell(data.get("student") or "-"), cell(data.get("subject") or "-"), cell(_num(max_marks)),
         Paragraph(f"<b>{_num(score)}</b>", st["cell"]), cell(f"{percent:.1f}%")],
    ]
    table = Table(rows, colWidths=[3.6 * cm, 4.2 * cm, 3.0 * cm, 3.2 * cm, 3.0 * cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("BACKGROUND", (0, 1), (-1, 1), LIGHT),
        ("BOX", (0, 0), (-1, -1), 0.6, NAVY),
        ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#C9D3E0")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story += [table, Spacer(1, 6)]

    def section(title, text):
        story.append(Paragraph(title, st["h"]))
        story.append(Paragraph(_txt(text) if str(text).strip() else "-", st["box"]))

    section("Question", data["question"])
    section("Marking rubric", data["rubric"])
    if str(data.get("reference") or "").strip():
        section("Reference answer", data["reference"])
    section("Student answer (text reviewed by teacher)", data["answer"])

    sim_lines = data.get("similarity_lines")
    if sim_lines:
        story.append(Paragraph("Copied-answer check", st["h"]))
        story.append(Paragraph(_txt("\n".join(sim_lines)), st["boxwarn"] if data.get("similarity_flag") else st["box"]))

    if image_bytes:
        try:
            story.append(KeepTogether([Paragraph("Uploaded handwritten answer", st["h"]),
                                       _image_flowable(image_bytes), Spacer(1, 6)]))
        except Exception:
            pass          # an unreadable image must never stop the report from being created

    section("AI feedback", data["feedback"])

    story.append(Spacer(1, 6))
    story.append(Paragraph(
        f"Model: {_txt(data.get('model', ''))}. Marks and feedback are generated by an AI model and "
        "may contain errors, and handwriting OCR can misread text. This report supports, but does not "
        "replace, the judgement of the teacher.", st["sub"]))

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buf.getvalue()


def save_report(pdf_bytes, reports_dir):
    """Save a copy in the reports folder and return its path."""
    os.makedirs(reports_dir, exist_ok=True)
    path = os.path.join(reports_dir, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
    with open(path, "wb") as f:
        f.write(pdf_bytes)
    return path