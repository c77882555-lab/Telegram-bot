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
    os.getenv("TELEGRAM_TOKEN")
    or os.getenv("BOT_TOKEN")
    or os.getenv("TELEGRAM_BOT_TOKEN")
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

MODEL_NAME = "openai/gpt-oss-120b"

if not BOT_TOKEN:
    raise RuntimeError("Missing TELEGRAM_TOKEN")

if not GROQ_API_KEY:
    raise RuntimeError("Missing GROQ_API_KEY")

client = Groq(
    api_key=GROQ_API_KEY,
    timeout=60.0,
    max_retries=1,
)

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

    logger.info(
        "Sending text to Groq (%s characters)",
        len(text),
    )

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

    if not result or not result.strip():
        raise ValueError("Groq returned an empty translation.")

    logger.info("Groq translation completed successfully.")

    return result.strip()


# ==========================================
# 3. استخراج النصوص من PDF
# ==========================================

def extract_blocks(page):
    blocks = page.get_text("blocks", sort=True)
    result = []

    for block in blocks:
        if len(block) < 5:
            continue

        text = block[4]

        if not isinstance(text, str):
            continue

        text = text.strip()

        if not text:
            continue

        result.append({"text": text})

    return result


# ==========================================
# 4. ترجمة جميع صفحات المحاضرة
# ==========================================

async def translate_document(source_path, status):
    translated_pages = []

    with fitz.open(source_path) as pdf:
        total_pages = len(pdf)

        if total_pages == 0:
            raise ValueError("ملف PDF فارغ.")

        total_paragraphs = 0

        for page in pdf:
            total_paragraphs += len(extract_blocks(page))

    if total_paragraphs == 0:
        raise ValueError(
            "ما لكيت نص قابل للاستخراج من الملف. "
            "يمكن المحاضرة عبارة عن صور ممسوحة ضوئياً."
        )

    logger.info(
        "PDF contains %s pages and %s paragraphs",
        total_pages,
        total_paragraphs,
    )

    completed = 0

    with fitz.open(source_path) as pdf:
        for page_index, page in enumerate(pdf, start=1):
            blocks = extract_blocks(page)
            translated_blocks = []

            logger.info(
                "Processing page %s/%s",
                page_index,
                total_pages,
            )

            for block_index, block in enumerate(
                blocks, start=1
            ):
                english_text = block["text"]

                logger.info(
                    "Starting page %s paragraph %s/%s",
                    page_index,
                    block_index,
                    len(blocks),
                )

                arabic_text = await asyncio.to_thread(
                    translate_text,
                    english_text,
                )

                translated_blocks.append({
                    "english": english_text,
                    "arabic": arabic_text,
                })

                completed += 1

                logger.info(
                    "Completed paragraph %s/%s",
                    completed,
                    total_paragraphs,
                )

                # تحديث حالة البوت كل 3 فقرات أو عند نهاية الصفحة
                if (
                    completed % 3 == 0
                    or block_index == len(blocks)
                ):
                    try:
                        await status.edit_text(
                            "🌐 جاري ترجمة المحاضرة...\n\n"
                            f"📄 الصفحة: {page_index}/{total_pages}\n"
                            f"📝 الفقرات المترجمة: "
                            f"{completed}/{total_paragraphs}"
                        )
                    except Exception:
                        logger.warning(
                            "Could not update progress message."
                        )

            translated_pages.append(translated_blocks)

    return translated_pages


# ==========================================
# 5. إنشاء PDF ثنائي اللغة
# ==========================================

def create_bilingual_pdf(
    translated_pages,
    translated_path,
):
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
        for page_number, blocks in enumerate(
            translated_pages, start=1
        ):
            parts = []

            for block in blocks:
                english = html.escape(block["english"])
                arabic = html.escape(block["arabic"])

                parts.append(
                    f"""
                    <div class="paragraph">
                        <div class="english">{english}</div>
                        <div class="arabic">{arabic}</div>
                    </div>
                    """
                )

            if not parts:
                parts.append(
                    "<p>No extractable text on this page.</p>"
                )

            html_content = f"""
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
                "Created translated page %s",
                page_number,
            )

    finally:
        writer.close()


# ==========================================
# 6. دمج PDF المترجم مع الأصلي
# ==========================================

def merge_pdfs(
    translated_path,
    source_path,
    output_path,
):
    translated = fitz.open(translated_path)
    original = fitz.open(source_path)
    output = fitz.open()

    try:
        output.insert_pdf(translated)
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

    logger.info("PDF merge completed.")


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

    logger.info(
        "Received PDF: %s | size=%s",
        filename,
        document.file_size,
    )

    status = await message.reply_text(
        "📥 استلمت المحاضرة!\n"
        "جاري تجهيز الملف..."
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

            # تنزيل الملف
            await status.edit_text(
                "📥 جاري تنزيل ملف PDF..."
            )

            tg_file = await context.bot.get_file(
                document.file_id
            )

            await tg_file.download_to_drive(
                source_path
            )

            with fitz.open(source_path) as pdf:
                page_count = len(pdf)

            if page_count == 0:
                raise ValueError("ملف PDF فارغ.")

            logger.info(
                "Downloaded PDF: %s pages",
                page_count,
            )

            # ترجمة النصوص
            await status.edit_text(
                f"📖 عدد الصفحات: {page_count}\n"
                "🌐 جاري استخراج النصوص وترجمتها..."
            )

            translated_pages = await translate_document(
                source_path,
                status,
            )

            # إنشاء PDF المترجم
            await status.edit_text(
                "📝 اكتملت الترجمة!\n"
                "جاري إنشاء ملف PDF..."
            )

            await asyncio.to_thread(
                create_bilingual_pdf,
                translated_pages,
                translated_path,
            )

            # دمج الملفين
            await status.edit_text(
                "📚 جاري إرفاق صفحات المحاضرة الأصلية..."
            )

            await asyncio.to_thread(
                merge_pdfs,
                translated_path,
                source_path,
                output_path,
            )

            if not os.path.exists(output_path):
                raise FileNotFoundError(
                    "لم يتم إنشاء ملف PDF النهائي."
                )

            if os.path.getsize(output_path) == 0:
                raise ValueError(
                    "الملف النهائي فارغ."
                )

            # إرسال الملف
            await status.edit_text(
                "✅ اكتملت المعالجة!\n"
                "جاري إرسال الملف..."
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

            logger.info(
                "PDF successfully sent to user."
            )

    except Exception as error:
        logger.exception(
            "PDF processing failed: %s",
            error,
        )

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
# 10. معالجة الأخطاء العامة
# ==========================================

async def error_handler(
    update: object,
    context: ContextTypes.DEFAULT_TYPE,
):
    logger.error(
        "Unhandled error: %s",
        context.error,
        exc_info=context.error,
    )


# ==========================================
# 11. تشغيل البوت
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

    app.add_error_handler(error_handler)

    logger.info("Bot is running...")

    app.run_polling()


if __name__ == "__main__":
    main()
