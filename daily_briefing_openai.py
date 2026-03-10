import os
import io
import json
import re
import base64
import requests
from datetime import datetime, timedelta, timezone

import pytz
from dotenv import load_dotenv

from openai import OpenAI

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as PlatypusImage,
)
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER

# ===================== CONFIG & CONSTANTS =====================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

client = OpenAI(api_key=OPENAI_API_KEY)

KST = pytz.timezone("Asia/Seoul")

# Map JSON keys to Display Names
REGION_MAP = {
    "seoul_metro": "수도권 (Seoul/Metro)",
    "gangwon": "강원도 (Gangwon)",
    "chungcheong": "충청권 (Chungcheong)",
    "jeolla": "전라권 (Jeolla)",
    "gyeongsang": "경상권 (Gyeongsang)",
    "jeju": "제주도 (Jeju)",
    "sea": "해상 (Marine)",
}

# ====== Auto-download Korean font for ReportLab ======
KOREAN_FONT_URL = (
    "https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf"
)
KOREAN_FONT_PATH = "NanumGothic.ttf"
FONT_NAME = "NanumGothic"


def register_korean_font():
    # 1. Check if font exists locally
    if not os.path.exists(KOREAN_FONT_PATH):
        print("Downloading Korean font (NanumGothic)...")
        try:
            resp = requests.get(KOREAN_FONT_URL, timeout=10)
            resp.raise_for_status()
            with open(KOREAN_FONT_PATH, "wb") as f:
                f.write(resp.content)
            print("Font downloaded successfully.")
        except Exception as e:
            print(f"Failed to download font: {e}")
            return False

    # 2. Register the font with ReportLab
    try:
        pdfmetrics.registerFont(TTFont(FONT_NAME, KOREAN_FONT_PATH))
        return True
    except Exception as e:
        print(f"Font registration failed: {e}")
        return False


HAS_KOREAN_FONT = register_korean_font()

# ===================== TIME & URL HELPERS =====================


def get_base_time_strings():
	"""
	Determine the base analysis time.

	The function uses the current date in KST and sets the base time
	to 00 UTC of the same calendar date.

	Example:
		2026-01-16 KST → base_utc = 2026-01-16 00:00 UTC
	"""
    now_kst = datetime.now(KST)
    base_utc = datetime(
        year=now_kst.year,
        month=now_kst.month,
        day=now_kst.day,
        tzinfo=timezone.utc,
    )
    ymd = base_utc.strftime("%Y%m%d")
    hhh = base_utc.strftime("%H")  # usually "00"
    return base_utc, ymd, hhh


def build_kma_urls(ymd, hhh):
	"""
	Construct KMA image URLs.

	Includes:
	- Satellite WV imagery
	- Surface analysis charts
	- 500 hPa geopotential height charts
	- 850 hPa wind charts

	Forecast steps: 0h, 12h, 24h, 36h, 48h.
	"""
    base_time = f"{ymd}{hhh}"  # e.g., 2026011600

    wv_url = (
        "https://www.weather.go.kr/w/repositary/image/sat/gk2a/EA/"
        f"gk2a_ami_le1b_wv063_ea020lc_{base_time}00.thn.png"
    )

    steps = ["s000", "s012", "s024", "s036", "s048"]

    surf_urls = [
        f"https://www.weather.go.kr/w/repositary/image/cht/img/"
        f"kim_gdps_erly_asia_surfce_ft06_pa4_{s}_{base_time}.png"
        for s in steps
    ]

    gph500_urls = [
        f"https://www.weather.go.kr/w/repositary/image/cht/img/"
        f"kim_gdps_erly_asia_gph500_ft06_pa4_{s}_{base_time}.png"
        for s in steps
    ]

    wnd850_urls = [
        f"https://www.weather.go.kr/w/repositary/image/cht/img/"
        f"kim_gdps_erly_asia_wnd850_ft06_pa4_{s}_{base_time}.png"
        for s in steps
    ]

    return {
        "wv": wv_url,
        "surface": surf_urls,
        "gph500": gph500_urls,
        "wnd850": wnd850_urls,
    }


def fetch_image(url, timeout=120):
	"""
	Download the image that will be embedded in the PDF.

	The image is returned as a BytesIO object. The same data will also be
	converted into a base64 data URL and sent to the OpenAI model,
	so the image only needs to be downloaded once.
	"""
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 200:
            return io.BytesIO(resp.content)
        else:
            print(f"Image download failed [{resp.status_code}]: {url}")
            return None
    except Exception as e:
        print(f"Exception while downloading image: {e} ({url})")
        return None


def bytesio_to_data_url(img_io, mime="image/png"):
	"""
	Convert a BytesIO image into a base64 data URL
	(data:image/png;base64,...).

	This avoids requiring the OpenAI server to fetch external URLs,
	which helps prevent timeouts when accessing KMA image servers.
	"""
    if img_io is None:
        return None
    img_io.seek(0)
    b64 = base64.b64encode(img_io.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


# ===================== OPENAI: JSON BRIEFING GENERATION =====================


def generate_briefing_json(base_utc, images):
    """
    Use OpenAI Responses API (multimodal) to analyze KMA charts
    using locally-downloaded images (BytesIO → base64 data URLs).
    Returns a Python dict with the briefing JSON.
    """
    valid_str = base_utc.strftime("%Y.%m.%d.%H UTC")
    kst_str = (base_utc + timedelta(hours=9)).strftime("%Y-%m-%d %H시")

    prompt_text = f"""
당신은 한국 기상청 수석 예보관입니다.
첨부된 위성영상(WV), 지상일기도(Surface), 500hPa, 850hPa 차트를 분석하여 일일 브리핑을 작성하세요.

Valid 시간: {valid_str} (KST: {kst_str})

아래 포맷에 맞춰 **한국어**로 작성해 주세요.
기상학적 전문 용어를 사용하되, 논리적 근거(Reasoning)를 명확히 하세요.

1. 종관 개황 (Synoptic overview)
2. 24–48시간 주요 특징 (Key features for 24–48h)
3. 한반도 체감 날씨 (수도권/강원/충청/전라/경상/제주/해상)
4. 위험 기상 요소 (Hazards - 강풍, 호우, 대설, 풍랑 등)
5. 주요 불확실성 (Uncertainties)
6. 내부 브리핑 요약 (3~5줄)

지상, 500hPa, 850hPa 차트는 각각 0h, 24h, 48h 예측장입니다. 시계열 변화를 분석에 반영하세요.

아래 JSON 스키마를 정확히 따르되, **JSON만** 출력하세요. (그 외 텍스트 금지)

{{
  "title": "한반도 일일 기상 브리핑",
  "synoptic_overview": "종관 개황 내용...",
  "key_features_24h": "24시간 예측 주요 특징...",
  "key_features_48h": "48시간 예측 주요 특징...",
  "sensible_weather": {{
      "seoul_metro": "수도권 날씨...",
      "gangwon": "강원도 날씨...",
      "chungcheong": "충청권...",
      "jeolla": "전라권...",
      "gyeongsang": "경상권...",
      "jeju": "제주도...",
      "sea": "해상..."
  }},
  "hazards": ["위험기상요소1: 주요 특징...", "위험기상요소2: 주요 특징..."],
  "uncertainties": "주요 불확실성...",
  "summary": "내부 브리핑 요약 (3~5줄)"
}}
"""

    content = [
        {"type": "input_text", "text": prompt_text},
    ]

    def add_img_io(img_io):
        data_url = bytesio_to_data_url(img_io)
        if data_url:
            content.append(
                {
                    "type": "input_image",
                    "image_url": data_url,
                }
            )

    # 1) Satellite WV
    add_img_io(images.get("wv"))

    # 2) Surface / 500 / 850 at 0h, 24h, 48h
    idx_list = [0, 2, 4]
    for idx in idx_list:
        add_img_io(images["surface"][idx])
        add_img_io(images["gph500"][idx])
        add_img_io(images["wnd850"][idx])

    print("Calling OpenAI for briefing JSON (with base64 images)...")
    response = client.responses.create(
        #model="gpt-4.1-mini",  # or "gpt-5.2-pro" if you want higher quality & cost
        model="gpt-5.2-pro",
        input=[
            {
                "role": "user",
                "content": content,
            }
        ],
        #response_format={"type": "json_object"},
        #temperature=0.3,
    )

    # Extract text from the response
    # (SDK structure: output -> content -> text.value)
    #out = response.output[0].content[0].text
    #raw = out.value if hasattr(out, "value") else str(out)
    raw = getattr(response, "output_text", None)

    #try:
    #    data = json.loads(raw)
    #except json.JSONDecodeError:
    #    print("❌ Failed to parse JSON from model. Raw preview:")
    #    print(raw[:500])
    #    data = {}
    return raw


# ===================== PDF BUILDING (STYLISH) =====================


def build_stylish_pdf(base_utc, urls, images, data) -> bytes:
    """
    Stylish single-PDF builder using ReportLab platypus.
    `images` is a dict of BytesIO objects (or None).
    `data` is the dict returned by generate_briefing_json().
    """
    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=15 * mm,
        rightMargin=15 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
    )

    styles = getSampleStyleSheet()
    font_main = FONT_NAME if HAS_KOREAN_FONT else "Helvetica"
    font_bold = FONT_NAME if HAS_KOREAN_FONT else "Helvetica-Bold"

    # Title Style
    style_title = ParagraphStyle(
        "BriefingTitle",
        parent=styles["Heading1"],
        fontName=font_bold,
        fontSize=20,
        leading=24,
        textColor=colors.navy,
        alignment=TA_CENTER,
        spaceAfter=10,
    )

    # Metadata Style
    style_meta = ParagraphStyle(
        "BriefingMeta",
        parent=styles["Normal"],
        fontName=font_main,
        fontSize=10,
        textColor=colors.gray,
        alignment=TA_CENTER,
        spaceAfter=20,
    )

    # Section Header Style
    style_h2 = ParagraphStyle(
        "SectionHeader",
        parent=styles["Heading2"],
        fontName=font_bold,
        fontSize=13,
        leading=16,
        textColor=colors.darkblue,
        spaceBefore=15,
        spaceAfter=8,
    )

    # Body Text Style
    style_body = ParagraphStyle(
        "BodyText",
        parent=styles["Normal"],
        fontName=font_main,
        fontSize=10,
        leading=15,
        alignment=TA_JUSTIFY,
        spaceAfter=5,
    )

    # Summary Box Style
    style_summary_box = ParagraphStyle(
        "SummaryText",
        parent=style_body,
        fontSize=11,
        leading=16,
        textColor=colors.black,
    )

    story = []

    # ----- HEADER -----
    title_text = data.get("title", "Daily Weather Briefing")
    valid_text = (
        f"Valid: {base_utc.strftime('%Y-%m-%d %H:00 UTC')}  |  Issued by OpenAI Forecast System"
    )

    story.append(Paragraph(title_text, style_title))
    story.append(Paragraph(valid_text, style_meta))

    # ----- SUMMARY BOX -----
    summary_content = data.get("summary", "").replace("\n", "<br/>")
    summary_para = Paragraph(f"<b>[요약]</b><br/>{summary_content}", style_summary_box)

    summary_table = Table([[summary_para]], colWidths=[170 * mm])
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.aliceblue),
                ("BOX", (0, 0), (-1, -1), 1, colors.steelblue),
                ("PADDING", (0, 0), (-1, -1), 10),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    story.append(summary_table)
    story.append(Spacer(1, 3 * mm))

    # ----- SYNOPTIC & KEY FEATURES -----
    story.append(Paragraph("1. Synoptic Overview & Key Features", style_h2))
    story.append(
        Paragraph(f"<b>[Synoptic]</b> {data.get('synoptic_overview', '-')}", style_body)
    )

    # ----- IMAGES (GRID) -----
    def prep_img(img_io, width=85 * mm, height=85 * mm):
        if img_io:
            try:
                img_io.seek(0)
                img = PlatypusImage(img_io, width=width, height=height)
                img.hAlign = "CENTER"
                return img
            except Exception:
                return Paragraph("(이미지 오류)", style_meta)
        return Paragraph("(No Image)", style_meta)

    # Row 1: Satellite & Surface 00h
    row1 = [
        [prep_img(images["wv"]), prep_img(images["surface"][0])],
        [
            Paragraph("GK2A Satellite (WV)", style_meta),
            Paragraph("Surface Analysis (00h)", style_meta),
        ],
    ]

    # Row 2: 500hPa 00h & 850hPa 00h
    row2 = [
        [prep_img(images["gph500"][0]), prep_img(images["wnd850"][0])],
        [
            Paragraph("500hPa Analysis (00h)", style_meta),
            Paragraph("850hPa Analysis (00h)", style_meta),
        ],
    ]

    img_table_data = row1 + row2
    t_img = Table(img_table_data, colWidths=[90 * mm, 90 * mm])
    t_img.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 2),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(t_img)

    story.append(Spacer(1, 3 * mm))
    story.append(
        Paragraph(f"<b>[24h Outlook]</b> {data.get('key_features_24h', '-')}", style_body)
    )

    # Row: 500hPa & Surface 24h
    row24 = [
        [prep_img(images["gph500"][2]), prep_img(images["surface"][2])],
        [
            Paragraph("500hPa Analysis (+24h)", style_meta),
            Paragraph("Surface Analysis (+24h)", style_meta),
        ],
    ]
    t_img24 = Table(row24, colWidths=[90 * mm, 90 * mm])
    t_img24.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 2),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(t_img24)

    story.append(Spacer(1, 3 * mm))
    story.append(
        Paragraph(f"<b>[48h Outlook]</b> {data.get('key_features_48h', '-')}", style_body)
    )

    # Row: 500hPa & Surface 48h
    row48 = [
        [prep_img(images["gph500"][4]), prep_img(images["surface"][4])],
        [
            Paragraph("500hPa Analysis (+48h)", style_meta),
            Paragraph("Surface Analysis (+48h)", style_meta),
        ],
    ]
    t_img48 = Table(row48, colWidths=[90 * mm, 90 * mm])
    t_img48.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 2),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(t_img48)

    # ----- HAZARDS -----
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph("2. Hazards & Warnings", style_h2))

    hazards = data.get("hazards", [])
    if hazards:
        table_data = []
        for h in hazards:
            if ":" in h:
                head, body = h.split(":", 1)
                table_data.append(
                    [
                        Paragraph(f"<b>{head}</b>", style_body),
                        Paragraph(body.strip(), style_body),
                    ]
                )
            else:
                table_data.append(
                    [
                        Paragraph("", style_body),
                        Paragraph(h, style_body),
                    ]
                )

        t_haz = Table(table_data, colWidths=[40 * mm, 130 * mm])
        t_haz.setStyle(
            TableStyle(
                [
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                    ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("PADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        story.append(t_haz)
    else:
        story.append(Paragraph("No significant hazards reported.", style_body))

    # ----- REGIONAL WEATHER -----
    story.append(Paragraph("3. Regional Weather Details", style_h2))

    sensible = data.get("sensible_weather", {})
    table_data = []

    if isinstance(sensible, dict):
        for key, value in sensible.items():
            region_name = REGION_MAP.get(key, key.upper())
            table_data.append(
                [
                    Paragraph(f"<b>{region_name}</b>", style_body),
                    Paragraph(value, style_body),
                ]
            )

    if table_data:
        t_regional = Table(table_data, colWidths=[40 * mm, 130 * mm])
        t_regional.setStyle(
            TableStyle(
                [
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                    ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("PADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        story.append(t_regional)

    # ----- UNCERTAINTIES -----
    story.append(Paragraph("4. Uncertainties", style_h2))
    story.append(Paragraph(data.get("uncertainties", "-"), style_body))

    # Build document
    doc.build(story)

    final = io.BytesIO()
    final.write(buffer.getvalue())
    final.seek(0)
    return final.read()


# ===================== DISCORD UPLOAD =====================


def post_to_discord(pdf_bytes, base_utc, data):
    if not DISCORD_WEBHOOK_URL:
        print("Discord Webhook URL not set. Skipping upload.")
        return

    filename = f"KP_Daily_Briefing_{base_utc.strftime('%Y%m%d_00UTC')}_OpenAI.pdf"
    content = (
        f"Korea Peninsula Daily Briefing (Powered by OpenAI)\n"
        f"Valid: {base_utc.strftime('%Y-%m-%d %H UTC')} "
        f"(KST {(base_utc + timedelta(hours=9)).strftime('%Y-%m-%d %H시')})\n"
        f"[Summary] {data.get('summary', '')}"       
    )

    files = {
        "file": (filename, pdf_bytes, "application/pdf"),
    }
    data = {
        "content": content,
    }

    try:
        resp = requests.post(DISCORD_WEBHOOK_URL, data=data, files=files)
        resp.raise_for_status()
        print("✅ PDF sent to Discord.")
    except Exception as e:
        print(f"Failed to upload to Discord: {e}")

# ===================== PARSE JSON =====================

def clean_parse_json(text):
    """
    Safely parses JSON from Gemini output, handling both
    Markdown code blocks (```json ... ```) and raw JSON strings.
    """
    try:
        # 1. Try parsing directly (Best for 'response_mime_type="application/json"')
        return json.loads(text)
    except json.JSONDecodeError:
        pass  # If failed, try cleaning

    try:
        # 2. Extract JSON content using Regex (Handles ```json, ```, and plain text)
        # Looks for the first '{' and the last '}'
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            cleaned_text = match.group(0)
            return json.loads(cleaned_text)
    except (json.JSONDecodeError, AttributeError):
        pass

    # 3. Fallback: Return empty dict or raise specific error
    print(f"❌ JSON Parsing Failed. Raw text preview: {text[:100]}...")
    return {}
    
# ===================== MAIN =====================


def main():
    base_utc, ymd, hhh = get_base_time_strings()
    print(f"Target Time: {ymd}{hhh} (00UTC)")

    urls = build_kma_urls(ymd, hhh)

    print("Downloading images for PDF...")
    images = {
        "wv": fetch_image(urls["wv"]),
        "surface": [fetch_image(u) for u in urls["surface"]],
        "gph500": [fetch_image(u) for u in urls["gph500"]],
        "wnd850": [fetch_image(u) for u in urls["wnd850"]],
    }

    print("Generating analysis with OpenAI...")
    data = generate_briefing_json(base_utc, images)

    print("Building PDF...")
    pdf_bytes = build_stylish_pdf(base_utc, urls, images, clean_parse_json(data))

    post_to_discord(pdf_bytes, base_utc, clean_parse_json(data))    
    
    # Save locally as well
    pdf_filename = f"KP_Daily_Briefing_{base_utc.strftime('%Y%m%d_00UTC')}_OpenAI.pdf"
    with open(pdf_filename, "wb") as f:
        f.write(pdf_bytes)
    print(f"✅ PDF saved: {pdf_filename}")
  
if __name__ == "__main__":
    main()
