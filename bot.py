import os
import re
import asyncio
import tempfile
import logging
import html
import time

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

BATCH_SIZE = 4
MAX_RETRIES = 5

if not BOT_TOKEN:
    raise RuntimeError("Missing TELEGRAM_TOKEN")

if not GROQ_API_KEY:
    raise RuntimeError("Missing GROQ_API_KEY")

client = Groq(
    api_key=GROQ_API_KEY,
    timeout=120.0,
    max_retries=0,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


# ==========================================
# 2. الاتصال بـ Groq مع إعادة المحاولة
# ==========================================

def groq_request(messages):
    for attempt in range(MAX_RETRIES):
        try:
            return client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.1,
            )

        except Exception as error:
            status_code = getattr(error, "status_code", None)

            if status_code == 429 or status_code in (500, 502, 503, 504):
                wait_time = min(2 ** (attempt + 1), 30)

                logger.warning(
                    "Groq error %s. Retry %s/%s after %s seconds.",
                    status_code,
                    attempt + 1,
                    MAX_RETRIES,
                    wait_time,
                )

                if attempt == MAX_RETRIES - 1:
                    raise

                time.sleep(wait_time)

            else:
                raise

    raise RuntimeError("Groq request failed after retries.")


# ==========================================
# 3. ترجمة فقرة واحدة
# ==========================================

def translate_text(text: str) -> str:
    if not text.strip():
        return ""

    response = groq_request([
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
    ])

    result = response.choices[0].message.content

    if not result or not result.strip():
        raise ValueError("Groq returned an empty translation.")

    return result.strip()


# ==========================================
# 4. ترجمة عدة فقرات بطلب واحد
# ==========================================

def translate_batch(texts):
    if not texts:
        return []

    if len(texts) == 1:
        return [translate_text(texts[0])]

    markers = [
        f"<<<BLOCK_{i}>>>"
        for i in range(1, len(texts) + 1)
    ]

    combined_text = "\n\n".join(
        f"{markers[i]}\n{text}"
        for i, text in enumerate(texts)
    )

    logger.info(
        "Sending batch of %s paragraphs to Groq.",
        len(texts),
    )

    response = groq_request([
        {
            "role": "system",
            "content": (
                "You are a professional medical translator. "
                "Translate each English block into accurate Arabic. "
                "Preserve medical terminology, numbers, abbreviations, "
                "lists, and all important details. Do not summarize. "
                "Return every block in the same order. "
                "Reproduce each block marker exactly as given, "
                "on a separate line, followed by its Arabic translation. "
                "Do not omit, rename, or add markers. "
                "Do not include explanations."
            ),
        },
        {
            "role": "user",
            "content": combined_text,
        },
    ])

    result = response.choices[0].message.content

    if not result or not result.strip():
        raise ValueError("Groq returned an empty batch translation.")

    pattern = re.compile(r"<<<BLOCK_(\d+)>>>")
    matches = list(pattern.finditer(result))

    translations = {}

    for index, match in enumerate(matches):
        block_number = int(match.group(1))
        start = match.end()

        end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(result)
        )

        translations[block_number] = result[start:end].strip()

    output = []

    for i, text in enumerate(texts, start=1):
        translated = translations.get(i, "")

        if not translated:
            logger.warning(
                "Missing translation for block %s. Retrying individually.",
                i,
            )
            translated = translate_text(text)

        output.append(translated)

    logger.info(
        "Batch translation completed: %s paragraphs.",
        len(output),
    )

    return output


# ==========================================
# 5. استخراج النصوص من PDF
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
# 6. ترجمة جميع صفحات المحاضرة
# ==========================================

async def translate_document(source_path, status):
    translated_pages = []

    with fitz.open(source_path) as pdf:
        total_pages = len(pdf)

        if total_pages == 0:
            raise ValueError("ملف PDF فارغ.")

        all_blocks = [
            extract_blocks(page)
            for page in pdf
        ]

    total_paragraphs = sum(
        len(blocks) for blocks in all_blocks
    )

    if total_paragraphs == 0:
        raise ValueError(
            "ما لكيت نص قابل للاستخراج من الملف. "
            "يمكن المحاضرة عبارة عن صور ممسوحة ضوئياً."
        )

    logger.info(
        "PDF contains %s pages and %s paragraphs.",
        total_pages,
        total_paragraphs,
    )

    completed = 0

    for page_index, blocks in enumerate(all_blocks, start=1):
        translated_blocks = []

        logger.info(
            "Processing page %s/%s.",
            page_index,
            total_pages,
        )

        for start in range(0, len(blocks), BATCH_SIZE):
            batch = blocks[start:start + BATCH_SIZE]

            english_texts = [
                block["text"] for block in batch
            ]

            logger.info(
                "Translating page %s, batch %s.",
                page_index,
                start // BATCH_SIZE + 1,
            )

            arabic_texts = await asyncio.to_thread(
                translate_batch,
                english_texts,
            )

            for english, arabic in zip(english_texts, arabic_texts):
                translated_blocks.append({
                    "english": english,
                    "arabic": arabic,
                })

            completed += len(batch)

            logger.info(
                "Completed %s/%s paragraphs.",
                completed,
                total_paragraphs,
            )

            try:
                await status.edit_text(
                    "🌐 جاري ترجمة المحاضرة...\n\n"
                    f"📄 الصفحة: {page_index}/{total_pages}\n"
                    f"📝 الفقرات المترجمة: "
                    f"{completed}/{total_paragraphs}"
                )
            except Exception:
                logger.warning("Could not update progress message.")

        translated_pages.append(translated_blocks)

    return translated_pages


# ==========================================
# 7. إنشاء PDF ثنائي اللغة
# ==========================================

def create_bilingual_pdf(translated_pages, translated_path):
    logger.info("Starting PDF creation.")

    writer = fitz.DocumentWriter(translated_path)

    page_width = 595
    page_height = 842

    page_rect = fitz.Rect(0, 0, page_width, page_height)

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
        for page_number, blocks in enumerate(translated_pages, start=1):
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
                parts.append("<p>No extractable text on this page.</p>")

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
            generated_pages = 0

            while more:
                generated_pages += 1

                if generated_pages > 100:
                    raise RuntimeError(
                        f"Too many generated pages for source page "
                        f"{page_number}. Stopping to prevent a loop."
                    )

                device = writer.begin_page(page_rect)

                # التصحيح الأساسي: place ترجع (more, filled)
                more, filled = story.place(content_rect)

                story.draw(device)
                writer.end_page()

                logger.info(
                    "PDF source page %s: generated page %s.",
                    page_number,
                    generated_pages,
                )

            logger.info("Created translated page %s.", page_number)

    finally:
        writer.close()

    if not os.path.exists(translated_path):
        raise FileNotFoundError("Translated PDF was not created.")

    if os.path.getsize(translated_path) == 0:
        raise ValueError("Translated PDF is empty.")

    logger.info(
        "Translated PDF created successfully. Size: %s bytes.",
        os.path.getsize(translated_path),
    )


# ==========================================
# 8. دمج PDF المترجم مع الأصلي
# ==========================================

def merge_pdfs(translated_path, source_path, output_path):
    logger.info("Starting PDF merge.")

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

    if not os.path.exists(output_path):
        raise FileNotFoundError("Final PDF was not created.")

    if os.path.getsize(output_path) == 0:
        raise ValueError("Final PDF is empty.")

    logger.info(
        "PDF merge completed. Final size: %s bytes.",
        os.path.getsize(output_path),
    )


# ==========================================
# 9. أمر /start
# ==========================================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "أهلاً بيك! 📚\n\n"
        "أرسل محاضرتك بصيغة PDF.\n\n"
        "راح أرتب النص الإنكليزي بالأسود، "
        "والترجمة العربية بالأحمر تحته مباشرةً، "
        "وبعدها أرفق صفحات المحاضرة الأصلية."
    )


# ==========================================
# 10. معالجة ملفات PDF
# ==========================================

async def handle_pdf(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    document = message.document

    if not document:
        return

    filename = document.file_name or "lecture.pdf"

    if not filename.lower().endswith(".pdf"):
        await message.reply_text("❌ أرسل ملف PDF فقط.")
        return

    logger.info(
        "Received PDF: %s | size=%s",
        filename,
        document.file_size,
    )

    status = await message.reply_text(
        "📥 استلمت المحاضرة!\nجاري تجهيز الملف..."
    )

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = os.path.join(temp_dir, "source.pdf")
            translated_path = os.path.join(temp_dir, "translated.pdf")
            output_path = os.path.join(temp_dir, "bilingual_lecture.pdf")

            # تنزيل الملف
            await status.edit_text("📥 جاري تنزيل ملف PDF...")

            tg_file = await context.bot.get_file(document.file_id)
            await tg_file.download_to_drive(source_path)

            with fitz.open(source_path) as pdf:
                page_count = len(pdf)

            if page_count == 0:
                raise ValueError("ملف PDF فارغ.")

            logger.info("Downloaded PDF: %s pages.", page_count)

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
                "📝 اكتملت الترجمة!\nجاري إنشاء ملف PDF..."
            )

            logger.info("PDF creation started.")

            await asyncio.to_thread(
                create_bilingual_pdf,
                translated_pages,
                translated_path,
            )

            logger.info("PDF creation finished.")

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

            logger.info("PDF merge finished.")

            if not os.path.exists(output_path):
                raise FileNotFoundError(
                    "لم يتم إنشاء ملف PDF النهائي."
                )

            output_size = os.path.getsize(output_path)

            if output_size == 0:
                raise ValueError("الملف النهائي فارغ.")

            logger.info(
                "Final PDF ready: %s bytes.",
                output_size,
            )

            # إرسال الملف
            await status.edit_text(
                "✅ اكتملت المعالجة!\nجاري إرسال الملف..."
            )

            logger.info("Starting Telegram document upload.")

            with open(output_path, "rb") as file:
                await context.bot.send_document(
                    chat_id=message.chat_id,
                    document=file,
                    filename="Bilingual_Lecture.pdf",
                    caption=(
                        "✅ تمت معالجة المحاضرة!\n\n"
                        "الإنكليزي بالأسود، "
                        "والترجمة العربية بالأحمر تحته، "
                        "وصفحات المحاضرة الأصلية مرفقة بالنهاية."
                    ),
                    connect_timeout=30,
                    read_timeout=180,
                    write_timeout=180,
                    pool_timeout=30,
                )

            logger.info("PDF successfully sent to user.")
            await status.delete()

    except Exception as error:
        logger.exception("PDF processing failed: %s", error)

        error_text = str(error)[:700]

        try:
            await status.edit_text(
                "❌ صار خطأ أثناء معالجة الملف:\n\n"
                f"{error_text}"
            )
        except Exception:
            await message.reply_text(f"❌ خطأ: {error_text}")


# ==========================================
# 11. أمر /help
# ==========================================

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📚 طريقة الاستخدام:\n\n"
        "1. أرسل ملف PDF.\n"
        "2. انتظر اكتمال الترجمة.\n"
        "3. استلم ملف المحاضرة المترجم."
    )


# ==========================================
# 12. معالجة الأخطاء العامة
# ==========================================

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.error(
        "Unhandled error: %s",
        context.error,
        exc_info=context.error,
    )


# ==========================================
# 13. تشغيل البوت
# ==========================================

def main():
    app = (
        Application.builder()
        .token(BOT_TOKEN)
        .build()
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))

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
