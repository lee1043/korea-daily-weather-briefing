import os
import io
import requests
from google import genai
from google.genai import types
import time
from datetime import datetime, timedelta, timezone
import pytz
from dotenv import load_dotenv
import emoji
import markdown
import textwrap
import re
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
import unicodedata

# ====== Environment Configuration ======
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DISCORD_NEWS_WEBHOOK_URL = os.getenv("DISCORD_NEWS_WEBHOOK_URL")

KOREAN_FONT_NAME = "NanumGothic"
EMOJI_FONT_NAME = "NotoEmoji"

pdfmetrics.registerFont(TTFont(KOREAN_FONT_NAME, "NanumGothic.ttf"))
pdfmetrics.registerFont(TTFont(EMOJI_FONT_NAME, "NotoEmoji.ttf"))

KST = pytz.timezone("Asia/Seoul")

# ----- Generate date strings based on today's 00 UTC -----
def get_base_time_strings():
    now_kst = datetime.now(KST)
    base_utc = datetime(
        year=now_kst.year,
        month=now_kst.month,
        day=now_kst.day,
        tzinfo=timezone.utc,
    )
    ymd = base_utc.strftime("%Y%m%d")
    hhh = base_utc.strftime("%H")
    return base_utc, ymd, hhh
    
def get_weather_news():
    print("🔍 Searching for weather news...")
    
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    # Prompt: We ask for the summary, but we don't ask it to write the URLs.
    # We will attach the URLs ourselves from the metadata to ensure they are real.
    prompt = (
        "Search for the most impactful worldwide weather news from the last 24 hours. "
        "Select the top 3-5 major events. "
        "Write a short, engaging summary for Discord in Markdown. "
        "Do not invent URLs. Just write the news summaries with bold headlines. "
        "Start with a greeting and the current date. "        
        "Write in Korean, please."
    )

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                response_modalities=["TEXT"]
            )
        )
        
        # 1. Get the Main Text
        main_text = response.text
        
        # 2. Extract REAL URLs from Grounding Metadata
        # This is the secret sauce to avoid fake links.
        sources_text = "\n\n**📚 Real Sources:**\n"
        
        # Check if we have grounding metadata
        if (response.candidates[0].grounding_metadata and 
            response.candidates[0].grounding_metadata.grounding_chunks):
            
            unique_links = set()
            
            for chunk in response.candidates[0].grounding_metadata.grounding_chunks:
                # In the new SDK, 'web' holds the source info
                if chunk.web and chunk.web.uri and chunk.web.title:
                    title = chunk.web.title
                    url = chunk.web.uri
                    
                    if url not in unique_links:
                        sources_text += f"- [{title}]({url})\n"
                        unique_links.add(url)
        else:
            sources_text += "(No specific source links returned by Google Search)"

        # Combine text + real links
        final_message = main_text + sources_text
        return final_message

    except Exception as e:
        print(f"Error: {e}")
        return None

def post_to_discord(content):
    """
    Splits long messages into chunks <= 2000 chars and sends them sequentially.
    """
    if not content: return

    # Discord limit
    LIMIT = 2000
    
    # If it fits in one message, just send it
    if len(content) <= LIMIT:
        requests.post(DISCORD_NEWS_WEBHOOK_URL, json={"content": content})
        print("✅ Posted to Discord (Single message)")
        return

    # --- SMART SPLITTING LOGIC ---
    print(f"⚠️ Content length ({len(content)}) exceeds limit. Splitting...")
    
    lines = content.split('\n')
    current_chunk = ""
    
    for line in lines:
        # Check if adding this line (plus a newline) would exceed the limit
        if len(current_chunk) + len(line) + 1 > LIMIT:
            # Send the current chunk
            requests.post(DISCORD_NEWS_WEBHOOK_URL, json={"content": current_chunk})
            print("   -> Sent part")
            
            # Reset chunk to the current line
            current_chunk = line + "\n"
            
            # Sleep briefly to be nice to Discord's servers
            time.sleep(1)
        else:
            # Add line to current chunk
            current_chunk += line + "\n"
            
    # Send any remaining text
    if current_chunk:
        requests.post(DISCORD_NEWS_WEBHOOK_URL, json={"content": current_chunk})
        print("   -> Sent final part")
    
    print("✅ All parts posted successfully!")

def is_emoji(ch: str) -> bool:
    cp = ord(ch)
    return (
        0x1F300 <= cp <= 0x1FAFF   # Misc Symbols and Pictographs, etc.
        or 0x2600 <= cp <= 0x26FF  # Misc symbols (☀, ☔, etc.)
        or 0x2700 <= cp <= 0x27BF  # Dingbats
    )

def draw_segment_with_emoji(c, x, y, text: str,
                            text_font: str, emoji_font: str,
                            font_size: int = 11) -> float:
    """
    Draw a single text segment (no newlines) at (x, y),
    using text_font for normal chars and emoji_font for emojis.
    Returns the WIDTH advanced (so caller can track cursor).
    """
    cursor_x = x
    buffer = ""

    for ch in text:
        if is_emoji(ch):
            # flush normal buffer
            if buffer:
                c.setFont(text_font, font_size)
                c.drawString(cursor_x, y, buffer)
                cursor_x += pdfmetrics.stringWidth(buffer, text_font, font_size)
                buffer = ""
            # draw emoji
            c.setFont(emoji_font, font_size)
            c.drawString(cursor_x, y, ch)
            cursor_x += pdfmetrics.stringWidth(ch, emoji_font, font_size)
        else:
            buffer += ch

    if buffer:
        c.setFont(text_font, font_size)
        c.drawString(cursor_x, y, buffer)
        cursor_x += pdfmetrics.stringWidth(buffer, text_font, font_size)

    return cursor_x - x  # total width advanced
                                
def draw_markdown_line_with_links(
    c,
    x,
    y,
    line: str,
    text_font: str,
    emoji_font: str,
    font_size: int = 11,
):
    """
    Draw a single line of markdown text:
      - normal text
      - [title](url) → only 'title' is visible, clickable via linkURL
    """
    cursor_x = x
    pos = 0
    pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')

    for match in pattern.finditer(line):
        start, end = match.span()
        title = match.group(1)
        url   = match.group(2)

        # text before link
        pre_text = line[pos:start]
        if pre_text:
            w = draw_segment_with_emoji(
                c, cursor_x, y, pre_text,
                text_font=text_font,
                emoji_font=emoji_font,
                font_size=font_size,
            )
            cursor_x += w

        # link text (visible)
        link_x_start = cursor_x
        w = draw_segment_with_emoji(
            c, link_x_start, y, title,
            text_font=text_font,
            emoji_font=emoji_font,
            font_size=font_size,
        )
        link_x_end = link_x_start + w

        # clickable rect
        c.linkURL(
            url,
            (link_x_start, y - 2, link_x_end, y + font_size),
            relative=0,
        )

        cursor_x = link_x_end
        pos = end

    # tail after last link
    tail = line[pos:]
    if tail:
        draw_segment_with_emoji(
            c, cursor_x, y, tail,
            text_font=text_font,
            emoji_font=emoji_font,
            font_size=font_size,
        )

def measure_text_width(text, text_font, emoji_font, font_size=11):
    from reportlab.pdfbase import pdfmetrics

    width = 0
    buffer = ""
    for ch in text:
        if is_emoji(ch):
            # flush buffer
            if buffer:
                width += pdfmetrics.stringWidth(buffer, text_font, font_size)
                buffer = ""
            width += pdfmetrics.stringWidth(ch, emoji_font, font_size)
        else:
            buffer += ch

    # flush buffer end
    if buffer:
        width += pdfmetrics.stringWidth(buffer, text_font, font_size)

    return width

def get_thumbnail_url(page_url: str, timeout: float = 5.0) -> str | None:
    """
    Try to fetch a representative thumbnail image for a web page.
    Priority:
      1. <meta property="og:image">
      2. <meta name="twitter:image">
      3. <link rel="icon"> or similar
      4. /favicon.ico as last resort
    Returns absolute image URL or None.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; WeatherNewsBot/1.0)"
    }

    try:
        resp = requests.get(page_url, headers=headers, timeout=timeout)
        resp.raise_for_status()
    except Exception as e:
        print(f"[thumb] Failed to fetch page {page_url}: {e}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    # 1. og:image
    og = soup.find("meta", property="og:image")
    if og and og.get("content"):
        return urljoin(page_url, og["content"])

    # 2. twitter:image
    tw = soup.find("meta", attrs={"name": "twitter:image"})
    if tw and tw.get("content"):
        return urljoin(page_url, tw["content"])


def draw_thumbnail_from_url(c,
                            image_url: str,
                            link_url: str,                            
                            x: float,
                            y: float,
                            size: float = 10,
                            timeout: float = 5.0):
    """
    Download an image and draw it at (x, y) with a square size.
    (x, y) = lower-left corner.
    """
    if not image_url:
        return

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; WeatherNewsBot/1.0)"
    }

    try:
        resp = requests.get(image_url, headers=headers, timeout=timeout)
        resp.raise_for_status()
    except Exception as e:
        print(f"[thumb] Failed to download image {image_url}: {e}")
        return

    try:
        img_bytes = io.BytesIO(resp.content)
        img_reader = ImageReader(img_bytes)
        w, h = img_reader.getSize()
        if size > h:
            size = h
        aspect = w / float(h)
        thumb_h = size
        thumb_w = size * aspect

        c.drawImage(
            img_reader,
            x=x,
            y=y,
            width=thumb_w,
            height=thumb_h,
            preserveAspectRatio=False,  # we already handled it
            mask="auto",
        )

        # Make it clickable
        if link_url:
            c.linkURL(
                link_url,
                (x, y, x + thumb_w, y + thumb_h),
                relative=0
            )                                
    except Exception as e:
        print(f"[thumb] Failed to draw image {image_url}: {e}")
        return
        
def generate_weather_news_pdf_from_markdown(content_md: str,
                                            base_utc: datetime | None = None) -> bytes:
    if base_utc is None:
        base_utc = datetime.utcnow()

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)

    width, height = A4
    margin_left = 20 * mm
    margin_right = 20 * mm
    margin_top = 25 * mm
    margin_bottom = 20 * mm
    line_height = 14  # base spacing

    # ------------------------------------
    # 1) Header bar (colored)
    # ------------------------------------
    header_height = 25 * mm

    # Header background
    c.setFillColorRGB(0.08, 0.16, 0.32)  # dark blue
    c.rect(0, height - header_height, width, header_height, stroke=0, fill=1)

    # Title text in header
    c.setFillColorRGB(1, 1, 1)
    header_title = "전세계 주요 기상 뉴스 요약"
    header_sub = base_utc.strftime("발행: %Y-%m-%d %H:%M UTC")

    # Slight padding inside header
    header_x = margin_left
    header_y = height - header_height + 8 * mm

    draw_segment_with_emoji(
        c, header_x, header_y + 8,
        "🌍 " + header_title,
        text_font=KOREAN_FONT_NAME,
        emoji_font=EMOJI_FONT_NAME,
        font_size=13,
    )
    draw_segment_with_emoji(
        c, header_x, header_y - 2,
        header_sub,
        text_font=KOREAN_FONT_NAME,
        emoji_font=EMOJI_FONT_NAME,
        font_size=9,
    )

    # Start text area below header
    c.setFillColorRGB(0, 0, 0)
    y = height - header_height - 10 * mm

    # ------------------------------------
    # text wrapping
    # ------------------------------------
    max_chars_per_line = 55
    wrapper = textwrap.TextWrapper(width=max_chars_per_line)

    in_sources_section = False  # flag when inside "📚 Real Sources"

    for raw_line in content_md.splitlines():
        line = raw_line.rstrip("\n")

        # Horizontal rule / blank → small space
        if line.strip() == "" or line.strip() == "---":
            y -= line_height * 0.7
            continue

        # --------------------------------------------------
        # “📚 Real Sources” heading → styled section label
        # --------------------------------------------------
        if line.strip().startswith("**📚") and "Real Sources" in line:
            in_sources_section = True

            # space before section
            y -= line_height * 0.5
            if y < margin_bottom:
                c.showPage()
                y = height - margin_top

            # draw label with accent color
            label_text = line.strip().strip("*")  # remove **…**
            c.setFillColorRGB(0.15, 0.35, 0.65)  # bluish
            draw_markdown_line_with_links(
                c,
                margin_left,
                y,
                label_text,
                text_font=KOREAN_FONT_NAME,
                emoji_font=EMOJI_FONT_NAME,
                font_size=12,
            )
            c.setFillColorRGB(0, 0, 0)
            y -= line_height * 1.2
            continue

        # --------------------------------------------------
        # H2-style headlines: any line that contains **bold**
        # --------------------------------------------------
        if "**" in line:
            # If there's at least one bold segment, treat the WHOLE line as H2.
            # Remove all **...** markers for rendering.
            if re.search(r"\*\*(.+?)\*\*", line):
                headline_text = re.sub(r"\*\*(.+?)\*\*", r"\1", line).strip()
            else:
                headline_text = line.strip()

            if headline_text:
                # spacing before headline
                y -= line_height * 0.5
                if y < margin_bottom:
                    c.showPage()
                    y = height - margin_top

                h2_font_size = 13

                # Wrap headline in case it’s long
                h2_segments = wrapper.wrap(headline_text) or [""]

                for seg in h2_segments:
                    # Height of this H2 bar
                    pad_x = 4 * mm
                    pad_y = 1.5 * mm
                    rect_h = h2_font_size + pad_y * 2

                    # Page break if needed
                    if y - rect_h < margin_bottom:
                        c.showPage()
                        y = height - margin_top

                    # Measure visible text width (Korean + emoji aware)
                    text_width = measure_text_width(
                        seg,
                        text_font=KOREAN_FONT_NAME,
                        emoji_font=EMOJI_FONT_NAME,
                        font_size=h2_font_size,
                    )

                    # Background rectangle geometry
                    rect_x = margin_left - pad_x
                    rect_y = y - pad_y
                    rect_w = text_width + pad_x * 2

                    # Draw background bar
                    c.setFillColorRGB(0.90, 0.95, 1.00)  # light blue
                    c.roundRect(rect_x, rect_y, rect_w, rect_h, radius=2 * mm,
                                stroke=0, fill=1)

                    # Draw headline text on top
                    c.setFillColorRGB(0.1, 0.1, 0.1)    # dark text
                    draw_markdown_line_with_links(
                        c,
                        margin_left,
                        y,
                        seg,
                        text_font=KOREAN_FONT_NAME,
                        emoji_font=EMOJI_FONT_NAME,
                        font_size=h2_font_size,
                    )

                    y -= rect_h + line_height * 0.2

                # Reset to body text color
                c.setFillColorRGB(0, 0, 0)
                y -= line_height * 0.1
                continue

        # --------------------------------------------------
        # Bullet list: - item / * item
        # --------------------------------------------------
        bullet_match = re.match(r"^\s*[-*]\s+(.*)$", line)
        if bullet_match:
            text = bullet_match.group(1)

            # If the bullet has a markdown link, avoid wrapping to keep URL hidden.
            if re.search(r'\[([^\]]+)\]\(([^)]+)\)', text):
                if y < margin_bottom:
                    c.showPage()
                    y = height - margin_top

                # Slight indent for bullets
                bullet_prefix = "• "

                # If we are in the Real Sources section, we’ll try to draw a thumbnail.
                if in_sources_section:
                    # Extract the first [title](url) from the bullet
                    m = re.search(r'\[([^\]]+)\]\(([^)]+)\)', text)
                    thumb_url = None
                    if m:
                        _, link_url = m.groups()
                        thumb_url = get_thumbnail_url(link_url)

                    thumb_size = 120  # pt
                    #thumb_margin = 6  # space between thumb and text

                    # Thumbnail position
                    thumb_x = margin_left + 4 * mm
                    # y is baseline; move image slightly below baseline
                    thumb_y = y - (thumb_size * 1.0) - 5

                    if thumb_url:
                        draw_thumbnail_from_url(
                            c,
                            thumb_url,
                            link_url,
                            x=thumb_x + 10,
                            y=thumb_y,
                            size=thumb_size,
                        )

                    # Text starts to the right of thumbnail
                    text_x = thumb_x

                    c.setFillColorRGB(0.2, 0.2, 0.2)  # slightly darker for sources
                    draw_markdown_line_with_links(
                        c,
                        text_x,
                        y,
                        bullet_prefix + text,  # "• [title](url)"
                        text_font=KOREAN_FONT_NAME,
                        emoji_font=EMOJI_FONT_NAME,
                        font_size=10,
                    )
                    c.setFillColorRGB(0, 0, 0)
                    if thumb_url:
                       y -= line_height + thumb_size + 10                    
                    else:                    
                       y -= line_height

                else:
                    # not in sources section → original behavior (no thumbnail)
                    draw_markdown_line_with_links(
                        c,
                        margin_left + 4 * mm,
                        y,
                        bullet_prefix + text,
                        text_font=KOREAN_FONT_NAME,
                        emoji_font=EMOJI_FONT_NAME,
                        font_size=11,
                    )
                    y -= line_height

            else:
                # bullet without link → normal wrapping
                wrapped = wrapper.wrap(text) or [""]
                for i, seg in enumerate(wrapped):
                    if y < margin_bottom:
                        c.showPage()
                        y = height - margin_top

                    prefix = "• " if i == 0 else "  "
                    draw_markdown_line_with_links(
                        c,
                        margin_left + 4 * mm,
                        y,
                        prefix + seg,
                        text_font=KOREAN_FONT_NAME,
                        emoji_font=EMOJI_FONT_NAME,
                        font_size=11,
                    )
                    y -= line_height
            continue

        # --------------------------------------------------
        # Normal paragraph line (may include links + emoji)
        # --------------------------------------------------
        for seg in wrapper.wrap(line) or [""]:
            if y < margin_bottom:
                c.showPage()
                y = height - margin_top

            draw_markdown_line_with_links(
                c,
                margin_left,
                y,
                seg,
                text_font=KOREAN_FONT_NAME,
                emoji_font=EMOJI_FONT_NAME,
                font_size=11,
            )
            y -= line_height

    c.showPage()
    c.save()

    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes

if __name__ == "__main__":
    news_update = get_weather_news()
    #print(news_update)
    if news_update:
        post_to_discord(news_update)         
    base_utc, ymd, hhh = get_base_time_strings()     
    pdf_filename = f"Daily_Weather_News_{base_utc.strftime('%Y%m%d_00UTC')}_Gemini.pdf"
    with open(pdf_filename, "wb") as f:
        f.write(generate_weather_news_pdf_from_markdown(news_update))
    print(f"✅ PDF saved: {pdf_filename}")      