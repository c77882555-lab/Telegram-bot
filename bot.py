import os
import asyncio
import tempfile
import logging
import html

import fitz
from groq import Groq
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# =========================
# إعدادات البوت
# =========================

BOT_TOKEN = os.environ["BOT_TOKEN"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]

MODEL_NAME = "openai/gpt-oss-120b"

client = Groq(api_key=GROQ_API_KEY)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# =========================
# الترجمة عبر Groq
# =========================

def translate_text(text: str) -> str:
    """ترجمة فقرة إنكليزية إلى العربية."""

    if not text.strip():
        return ""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a professional medical lecture translator. "
                    "Translate the provided English text into accurate, "
                    "clear Arabic. Preserve medical terminology and all "
                    "important details, numbers, abbreviations, and lists. "
                    "Do not summarize, omit, or add explanations. "
                    "Return only the Arabic translation."
                ),
            },
            {
                "role": "user",
                "content": text,
            },
        ],
        temperature=0.1,
    )

    return response.choices[0].message.content.strip()


# =========================
# استخراج فقرات الصفحة
# =========================

def extract_blocks(page):
    """استخراج النصوص وترتيبها من الأعلى إلى الأسفل."""

    blocks = page.get_text("blocks", sort=True)
    result = []

    for block in blocks:
        x0, y0, x1, y1, text = block[:5]

        # تجاهل كتل الصور والكتل الفارغة
        if not isinstance(text, str) or not text.strip():
            continue

        text = text.strip()

        # تجاهل النصوص القصيرة جداً التي غالباً تكون أرقام صفحات
        if len(text) < 2:
            continue

        result.append({
            "text": text,
            "x0": x0,
            "y0": y0,
            "x1": x1,
            "y1": y1,
        })

    return result


# =========================
# إنشاء HTML ثنائي اللغة
# =========================

def make_bilingual_html(blocks):
    """الإنكليزي بالأسود والترجمة العربية بالأحمر تحته."""

    parts = []

    for block in blocks:
        english = html.escape(block["text"])

        arabic = translate_text(block["text"])
        arabic = html.escape(arabic)

        parts.append(
            f"""
            <div class="paragraph">
                <div class="english">{english}</div>
                <div class="arabic">{arabic}</div>
            </div>
            """
        )

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
    </head>
    <body>
        <h1 class="heading">Bilingual Lecture / المحاضرة المترجمة</h1>
        {''.join(parts)}
    </body>
    </html>
    """


# =========================
# إنشاء صفحات الترجمة
# =========================

def create_bilingual_pdf(source_path, translated_path):
    """إنشاء صفحات ثنائية اللغة مع تقسيم تلقائي للصفحات."""

    source = fitz.open(source_path)

    writer = fitz.DocumentWriter(translated_path)

    page_width = 595
    page_height = 842

    page_rect = fitz.Rect(0, 0, page_width, page_height)
    content_rect = fitz.Rect(42, 42, page_width - 42, page_height - 42)

    css = """
    body {
        font-family: sans-serif;
        font-size: 11pt;
        line-height: 1.45;
        color: #111111;
    }

    .heading {
        font-size: 17pt;
        text-align: center;
        margin-bottom: 22pt;
        color: #222222;
    }

    .paragraph {
        margin-bottom: 18pt;
        padding-bottom: 8pt;
        border-bottom: 0.5pt solid #dddddd;
    }

    .english {
        color: #000000;
        font-size: 11pt;
        text-align: left;
        direction: ltr;
        margin-bottom: 8pt;
        white-space: pre-wrap;
    }

    .arabic {
        color: #d00000;
        font-size: 13pt;
        text-align: right;
        direction: rtl;
        margin-bottom: 4pt;
        white-space: pre-wrap;
    }
    """

    try:
        for page_number, page in enumerate(source, start=1):
            blocks = extract_blocks(page)

            if not blocks:
                blocks = [{"text": "(No extractable text on this page)"}]

            html_content = make_bilingual_html(blocks)

            story = fitz.Story(
                html=html_content,
                user_css=css,
            )

            more = True

            while more:
                device = writer.begin_page(page_rect)
                more = story.place(content_rect)
                story.draw(device)
                writer.end_page()

            logger.info(
                "Translated page %s of %s",
                page_number,
                len(source),
            )

    finally:
        writer.close()
        source.close()


# =========================
# دمج الترجمة مع الأصل
# =========================

def merge_pdfs(translated_path, source_path, output_path):
    """صفحات الترجمة أولاً ثم صفحات المحاضرة الأصلية."""

    translated = fitz.open(translated_path)
    original = fitz.open(source_path)
    output = fitz.open()

    output.insert_pdf(translated)
    output.insert_pdf(original)

    output.save(output_path, garbage=4, deflate=True)

    output.close()
    translated.close()
    original.close()


# =========================
# أوامر البوت
# =========================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "أهلاً بيك! 📚\n\n"
        "أرسل ملف محاضرة بصيغة PDF.\n"
        "راح أترجم النص الإنكليزي إلى العربية، "
        "وأضع الترجمة الحمراء تحت كل فقرة إنكليزية، "
        "وبعدها أرفق صفحات المحاضرة الأصلية."
    )


async def handle_pdf(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    message = update.message
    document = message.document

    if not document or not document.file_name.lower().endswith(".pdf"):
        await message.reply_text("أرسل ملف PDF فقط.")
        return

    status = await message.reply_text(
        "📥 استلمت المحاضرة، دا أجهزها وأترجمها...\n"
        "قد يستغرق هذا بعض الوقت حسب عدد الصفحات."
    )

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = os.path.join(temp_dir, "source.pdf")
            translated_path = os.path.join(temp_dir, "translated.pdf")
            output_path = os.path.join(temp_dir, "bilingual_lecture.pdf")

            tg_file = await context.bot.get_file(document.file_id)
            await tg_file.download_to_drive(source_path)

            # فحص الملف
            pdf = fitz.open(source_path)
            page_count = len(pdf)
            pdf.close()

            if page_count == 0:
                await status.edit_text("الملف فارغ.")
                return

            await status.edit_text(
                f"📖 عدد الصفحات: {page_count}\n"
                "جاري استخراج النصوص وترجمتها..."
            )

            # تشغيل المعالجة الثقيلة خارج حلقة البوت
            await asyncio.to_thread(
                create_bilingual_pdf,
                source_path,
                translated_path,
            )

            await asyncio.to_thread(
                merge_pdfs,
                translated_path,
                source_path,
                output_path,
            )

            await status.edit_text("📤 اكتملت الترجمة، جاري إرسال الملف...")

            with open(output_path, "rb") as file:
                await message.reply_document(
                    document=file,
                    filename="Bilingual_Lecture.pdf",
                    caption=(
                        "✅ اكتملت المحاضرة!\n\n"
                        "الإنكليزي بالأسود، والترجمة العربية بالأحمر، "
                        "وصفحات المحاضرة الأصلية مرفقة في نهاية الملف."
                    ),
                )

            await status.delete()

    except Exception as e:
        logger.exception("PDF processing failed")
        await status.edit_text(
            "❌ صار خطأ أثناء معالجة الملف.\n"
            f"التفاصيل: {str(e)[:800]}"
        )


async def help_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    await update.message.reply_text(
        "طريقة الاستخدام:\n"
        "1. أرسل ملف PDF.\n"
        "2. انتظر اكتمال الترجمة.\n"
        "3. استلم الملف الثنائي اللغة."
    )


# =========================
# تشغيل البوت
# =========================

def main():
    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(
        MessageHandler(
            filters.Document.ALL,
            handle_pdf,
        )
    )

    logger.info("Bot is running...")
    app.run_polling()


if __name__ == "__main__":
    main()
