import os
import re
import asyncio
import logging
import tempfile
import math
from pathlib import Path

import fitz
from google import genai
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# ==========================================
# CONFIGURATION
# ==========================================

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

MODEL_NAME = "gemini-3.8-flash"
CHUNK_SIZE = 6000

if not TELEGRAM_TOKEN or not GEMINI_API_KEY:
    raise RuntimeError("Missing TELEGRAM_TOKEN or GEMINI_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)


# ==========================================
# START COMMAND
# ==========================================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text(
        "أهلاً بك في بوت ترجمة المحاضرات الجامعية 📚\n\n"
        "أرسل ملف PDF وسأضيف الترجمة العربية فوق كل صفحة "
        "مع الاحتفاظ بالصفحة الأصلية أسفلها.\n\n"
        "📄 الناتج: ملف PDF مترجم."
    )


# ==========================================
# TEXT SPLITTING
# ==========================================

def split_text(text, max_chars=CHUNK_SIZE):
    text = text.strip()
    if not text:
        return []

    paragraphs = text.splitlines()
    chunks = []
    current = ""

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        while len(paragraph) > max_chars:
            if current:
                chunks.append(current)
                current = ""

            chunks.append(paragraph[:max_chars])
            paragraph = paragraph[max_chars:]

        if not paragraph:
            continue

        candidate = (
            current + "\n\n" + paragraph
            if current else paragraph
        )

        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = paragraph

    if current:
        chunks.append(current)

    return chunks


# ==========================================
# GEMINI TRANSLATION
# ==========================================

def translate_chunk(text, part, total):
    prompt = f"""
أنت مترجم أكاديمي متخصص في العلوم الطبية والتحليلات المرضية.

ترجم النص الإنجليزي التالي إلى العربية الفصحى بدقة علمية.

التعليمات:
- ترجم كل المعلومات دون اختصار أو تلخيص.
- حافظ على المصطلحات الطبية والأرقام والاختصارات.
- اكتب المصطلح الإنجليزي بين قوسين عند الحاجة.
- حافظ على ترتيب العناوين والقوائم قدر الإمكان.
- لا تضف معلومات من عندك.
- أخرج الترجمة فقط دون مقدمة أو خاتمة.
- هذا الجزء {part} من أصل {total}.

النص:
{text}
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )

    result = response.text
    if not result or not result.strip():
        raise ValueError("Empty Gemini response")

    return result.strip()


async def translate_text(text):
    chunks = split_text(text)

    if not chunks:
        return ""

    results = []

    for index, chunk in enumerate(chunks, start=1):
        translated = await asyncio.to_thread(
            translate_chunk,
            chunk,
            index,
            len(chunks),
        )
        results.append(translated)

    return "\n\n".join(results)


# ==========================================
# CREATE BILINGUAL PDF
# ==========================================

def estimate_translation_height(text, page_width):
    """
    Estimate the space needed for the Arabic translation.
    The original page will be placed below this area.
    """
    usable_width = max(page_width - 70, 200)
    chars_per_line = max(25, int(usable_width / 7.5))

    lines = 0
    for paragraph in text.splitlines():
        if paragraph.strip():
            lines += max(
                1,
                math.ceil(len(paragraph) / chars_per_line)
            )
        else:
            lines += 1

    return max(150, min(2500, 85 + lines * 19))


def add_translation_page(output_doc, source_doc, page_index, translated):
    original_page = source_doc[page_index]
    width = original_page.rect.width
    height = original_page.rect.height

    translation_height = estimate_translation_height(
        translated,
        width,
    )

    # Create a taller page: translation above, original below.
    new_page = output_doc.new_page(
        width=width,
        height=height + translation_height,
    )

    # Arabic translation area.
    translation_rect = fitz.Rect(
        30,
        25,
        width - 30,
        translation_height - 20,
    )

    html = f"""
    <div dir="rtl" style="
        font-family: sans-serif;
        text-align: right;
        font-size: 11pt;
        line-height: 1.6;
        color: #111111;
    ">
        <h2 style="text-align:center; font-size:15pt;">
            الترجمة العربية
        </h2>
        {''.join(
            '<p>' + paragraph.replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;') + '</p>'
            for paragraph in translated.splitlines()
            if paragraph.strip()
        )}
    </div>
    """

    new_page.insert_htmlbox(
        translation_rect,
        html,
        css="p { margin: 3px 0; }",
        scale_low=0.5,
    )

    # Insert original page unchanged below the translation.
    original_rect = fitz.Rect(
        0,
        translation_height,
        width,
        translation_height + height,
    )

    new_page.show_pdf_page(
        original_rect,
        source_doc,
        page_index,
    )


# ==========================================
# DOCUMENT HANDLER
# ==========================================

async def handle_document(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    message = update.effective_message
    document = message.document

    if not document:
        return

    filename = document.file_name or "lecture.pdf"

    if not filename.lower().endswith(".pdf"):
        await message.reply_text(
            "❌ أرسل ملف PDF فقط."
        )
        return

    progress = await message.reply_text(
        "📥 استلمت المحاضرة.\n"
        "جاري قراءة الصفحات واستخراج النص..."
    )

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, "input.pdf")
            output_path = os.path.join(temp_dir, "translated.pdf")

            telegram_file = await document.get_file()
            await telegram_file.download_to_drive(input_path)

            source_doc = fitz.open(input_path)
            output_doc = fitz.open()

            total_pages = len(source_doc)

            for index, page in enumerate(source_doc):
                page_text = page.get_text("text").strip()

                if not page_text:
                    translated = (
                        "لم يتم العثور على نص قابل للاستخراج "
                        "في هذه الصفحة. قد تكون الصفحة صورة "
                        "وتحتاج إلى OCR."
                    )
                else:
                    translated = await translate_text(page_text)

                add_translation_page(
                    output_doc,
                    source_doc,
                    index,
                    translated,
                )

                await progress.edit_text(
                    "⏳ جاري ترجمة المحاضرة...\n\n"
                    f"تمت معالجة الصفحة {index + 1} "
                    f"من {total_pages}."
                )

            output_doc.save(output_path)
            output_doc.close()
            source_doc.close()

            with open(output_path, "rb") as result_file:
                await message.reply_document(
                    document=result_file,
                    filename=f"Translated_{Path(filename).stem}.pdf",
                    caption=(
                        "✅ اكتملت ترجمة المحاضرة!\n"
                        "📄 الترجمة العربية بالأعلى "
                        "والصفحة الأصلية بالأسفل."
                    ),
                )

            await progress.edit_text(
                "✅ تمت ترجمة المحاضرة وإرسال ملف PDF."
            )

    except Exception:
        logger.exception("PDF translation failed")
        await progress.edit_text(
            "❌ حدث خطأ أثناء ترجمة الملف.\n"
            "تحقق من إعدادات Gemini API وحاول مرة أخرى."
        )


# ==========================================
# MAIN
# ==========================================

def main():
    application = (
        Application.builder()
        .token(TELEGRAM_TOKEN)
        .build()
    )

    application.add_handler(
        CommandHandler("start", start)
    )

    application.add_handler(
        MessageHandler(
            filters.Document.ALL,
            handle_document,
        )
    )

    logger.info("PDF translation bot started")
    application.run_polling()


if __name__ == "__main__":
    main()
