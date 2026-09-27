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

# ==========================================
# 1. إعدادات البوت
# ==========================================

BOT_TOKEN = (
    os.getenv("BOT_TOKEN")
    or os.getenv("TELEGRAM_BOT_TOKEN")
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

MODEL_NAME = "openai/gpt-oss-120b"

if not BOT_TOKEN:
    raise RuntimeError(
        "Missing bot token. Set BOT_TOKEN or TELEGRAM_BOT_TOKEN in Render."
    )

if not GROQ_API_KEY:
    raise RuntimeError(
        "Missing GROQ_API_KEY. Add it to Render Environment Variables."
    )

client = Groq(api_key=GROQ_API_KEY)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


# ==========================================
# 2. ترجمة النص باستخدام Groq
# ==========================================

def translate_text(text: str) -> str:
    if not text.strip():
        return ""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a professional medical translator. "
                    "Translate the English text into accurate, clear Arabic. "
                    "Preserve medical terminology, numbers, abbreviations, "
                    "lists, and all important details. "
                    "Do not summarize or omit anything. "
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

    result = response.choices[0].message.content

    if not result:
        raise ValueError("Groq returned an empty translation.")

    return result.strip()


# ==========================================
# 3. استخراج النصوص من صفحات PDF
# ==========================================

def extract_blocks(page):
    blocks = page.get_text("blocks", sort=True)
    result = []

    for block in blocks:
        if len(block) < 5:
            continue

        x0, y0, x1, y1, text = block[:5]

        if not isinstance(text, str):
            continue

        text = text.strip()

        if not text:
            continue

        result.append({
            "text": text,
            "x0": x0,
            "y0": y0,
            "x1": x1,
            "y1": y1,
        })

    return result


# ==========================================
# 4. إنشاء HTML ثنائي اللغة
# ==========================================

def make_bilingual_html(blocks, page_number):
    parts = []

    for index, block in enumerate(blocks, start=1):
        english_text = block["text"]

        # ترجمة كل فقرة بشكل مستقل
        arabic_text = translate_text(english_text)

        english = html.escape(english_text)
        arabic = html.escape(arabic_text)

        parts.append(
            f"""
            <div class="paragraph">
                <div class="english">{english}</div>
                <div class="arabic">{arabic}</div>
            </div>
            """
        )

        logger.info(
            "Page %s: translated paragraph %s/%s",
            page_number,
            index,
            len(blocks),
        )

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
    </head>
    <body>
        <h1 class="heading">
            Lecture Translation - Page {page_number}
        </h1>
        {''.join(parts)}
    </body>
    </html>
    """


# ==========================================
# 5. إنشاء PDF مترجم
# ==========================================

def create_bilingual_pdf(source_path, translated_path):
    source = fitz.open(source_path)
    writer = fitz.DocumentWriter(translated_path)

    page_width = 595
    page_height = 842

    page_rect = fitz.Rect(
        0, 0, page_width, page_height
    )

    content_rect = fitz.Rect(
        42, 42, page_width - 42, page_height - 42
    )

    css = """
    body {
        font-family: sans-serif;
        font-size: 11pt;
        line-height: 1.5;
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
        padding-bottom: 10pt;
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
        total_pages = len(source)

        for page_number, page in enumerate(source, start=1):
            blocks = extract_blocks(page)

            if not blocks:
                blocks = [{
                    "text": "No extractable text on this page."
                }]

            logger.info(
                "Processing page %s/%s",
                page_number,
                total_pages,
            )

            html_content = make_bilingual_html(
                blocks,
                page_number,
            )

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

    finally:
        writer.close()
        source.close()


# ==========================================
# 6. دمج الترجمة مع صفحات المحاضرة الأصلية
# ==========================================

def merge_pdfs(translated_path, source_path, output_path):
    translated = fitz.open(translated_path)
    original = fitz.open(source_path)
    output = fitz.open()

    try:
        # صفحات الترجمة أولاً
        output.insert_pdf(translated)

        # صفحات المحاضرة الأصلية بعدها
        output.insert_pdf(original)

        output.save(
            output_path,
            garbage=4,
            deflate=True,
        )

    finally:
        output.close()
        translated.close()
        original.close()


# ==========================================
# 7. أمر /start
# ==========================================

async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    await update.message.reply_text(
        "أهلاً بيك! 📚\n\n"
        "أرسل محاضرتك بصيغة PDF.\n\n"
        "راح أرتب النص الإنكليزي بالأسود، "
        "والترجمة العربية بالأحمر تحته مباشرةً، "
        "وبعدها أرفق صفحات المحاضرة الأصلية."
    )


# ==========================================
# 8. معالجة ملفات PDF
# ==========================================

async def handle_pdf(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    message = update.message
    document = message.document

    if not document:
        return

    filename = document.file_name or "lecture.pdf"

    if not filename.lower().endswith(".pdf"):
        await message.reply_text(
            "❌ أرسل ملف PDF فقط."
        )
        return

    status = await message.reply_text(
        "📥 استلمت المحاضرة!\n"
        "جاري تجهيز الملف والترجمة..."
    )

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = os.path.join(
                temp_dir, "source.pdf"
            )

            translated_path = os.path.join(
                temp_dir, "translated.pdf"
            )

            output_path = os.path.join(
                temp_dir, "bilingual_lecture.pdf"
            )

            # تحميل الملف
            tg_file = await context.bot.get_file(
                document.file_id
            )

            await tg_file.download_to_drive(
                source_path
            )

            # فحص الملف
            with fitz.open(source_path) as pdf:
                page_count = len(pdf)

            if page_count == 0:
                await status.edit_text(
                    "❌ الملف فارغ."
                )
                return

            await status.edit_text(
                f"📖 عدد الصفحات: {page_count}\n"
                "جاري استخراج النصوص وترجمتها..."
            )

            # إنشاء صفحات الترجمة
            await asyncio.to_thread(
                create_bilingual_pdf,
                source_path,
                translated_path,
            )

            # دمج صفحات الترجمة والأصل
            await asyncio.to_thread(
                merge_pdfs,
                translated_path,
                source_path,
                output_path,
            )

            await status.edit_text(
                "✅ اكتملت المعالجة، جاري إرسال الملف..."
            )

            with open(output_path, "rb") as file:
                await message.reply_document(
                    document=file,
                    filename="Bilingual_Lecture.pdf",
                    caption=(
                        "✅ تمت معالجة المحاضرة!\n\n"
                        "الإنكليزي بالأسود، "
                        "والترجمة العربية بالأحمر تحته، "
                        "وصفحات المحاضرة الأصلية مرفقة بالنهاية."
                    ),
                )

            await status.delete()

    except Exception as error:
        logger.exception("PDF processing failed")

        error_text = str(error)[:700]

        try:
            await status.edit_text(
                "❌ صار خطأ أثناء معالجة الملف:\n\n"
                f"{error_text}"
            )
        except Exception:
            await message.reply_text(
                f"❌ خطأ: {error_text}"
            )


# ==========================================
# 9. أمر /help
# ==========================================

async def help_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    await update.message.reply_text(
        "📚 طريقة الاستخدام:\n\n"
        "1. أرسل ملف PDF.\n"
        "2. انتظر اكتمال الترجمة.\n"
        "3. استلم ملف المحاضرة المترجم."
    )


# ==========================================
# 10. تشغيل البوت
# ==========================================

def main():
    app = (
        Application.builder()
        .token(BOT_TOKEN)
        .build()
    )

    app.add_handler(
        CommandHandler("start", start)
    )

    app.add_handler(
        CommandHandler("help", help_command)
    )

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
