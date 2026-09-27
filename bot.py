import os
import re
import asyncio
import tempfile
import logging
import time

import fitz
import arabic_reshaper
from bidi.algorithm import get_display
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

ARABIC_FONT = os.getenv("ARABIC_FONT", "NotoNaskhArabic-Regular.ttf")

if not BOT_TOKEN:
    raise RuntimeError("Missing TELEGRAM_TOKEN (or BOT_TOKEN)")
if not GROQ_API_KEY:
    raise RuntimeError("Missing GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY, timeout=120.0, max_retries=0)

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
                    status_code, attempt + 1, MAX_RETRIES, wait_time,
                )
                if attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(wait_time)
            else:
                raise
    raise RuntimeError("Groq request failed after retries.")


# ==========================================
# 3. الترجمة
# ==========================================

def translate_text(text: str) -> str:
    if not text.strip():
        return ""
    response = groq_request([
        {
            "role": "system",
            "content": (
                "You are a professional translator. Translate the English text "
                "into accurate, clear Arabic. Preserve technical/medical terms, "
                "numbers, abbreviations, lists, and all details. Do not summarize. "
                "Return only the Arabic translation."
            ),
        },
        {"role": "user", "content": text},
    ])
    result = response.choices[0].message.content
    if not result or not result.strip():
        raise ValueError("Groq returned an empty translation.")
    return result.strip()


def translate_batch(texts):
    if not texts:
        return []
    if len(texts) == 1:
        return [translate_text(texts[0])]

    markers = [f"<<<BLOCK_{i}>>>" for i in range(1, len(texts) + 1)]
    combined_text = "\n\n".join(
        f"{markers[i]}\n{text}" for i, text in enumerate(texts)
    )
    response = groq_request([
        {
            "role": "system",
            "content": (
                "Translate each English block into accurate Arabic. Preserve "
                "terminology, numbers, abbreviations, lists, and all details. "
                "Return every block in the same order. Reproduce each marker "
                "exactly on a separate line, followed by its Arabic translation. "
                "Do not omit or add markers and do not include explanations."
            ),
        },
        {"role": "user", "content": combined_text},
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
        end = matches[index + 1].start() if index + 1 < len(matches) else len(result)
        translations[block_number] = result[start:end].strip()

    output = []
    for i, text in enumerate(texts, start=1):
        translated = translations.get(i, "")
        if not translated:
            logger.warning("Missing translation for block %s; retrying individually.", i)
            translated = translate_text(text)
        output.append(translated)
    return output


# ==========================================
# 4. استخراج الفقرات مع مواقعها
# ==========================================

def extract_blocks(page):
    raw_blocks = page.get_text("blocks", sort=True)
    result = []
    for block in raw_blocks:
        if len(block) < 5:
            continue
        text = block[4]
        if not isinstance(text, str):
            continue
        text = text.strip()
        if not text:
            continue

        x0, y0, x1, y1 = map(float, block[:4])
        # استبعاد كتل صغيرة جداً أو غير صالحة
        if x1 <= x0 or y1 <= y0:
            continue
        result.append({
            "text": text,
            "bbox": [x0, y0, x1, y1],
        })
    return result


async def translate_document(source_path, status):
    translated_pages = []
    with fitz.open(source_path) as pdf:
        total_pages = len(pdf)
        if total_pages == 0:
            raise ValueError("ملف PDF فارغ.")
        all_blocks = [extract_blocks(page) for page in pdf]

    total_paragraphs = sum(len(blocks) for blocks in all_blocks)
    if total_paragraphs == 0:
        raise ValueError(
            "ما لكيت نص قابل للاستخراج من الملف. "
            "يمكن المحاضرة عبارة عن صور ممسوحة ضوئياً."
        )

    completed = 0
    for page_index, blocks in enumerate(all_blocks, start=1):
        translated_blocks = []
        for start in range(0, len(blocks), BATCH_SIZE):
            batch = blocks[start:start + BATCH_SIZE]
            english_texts = [block["text"] for block in batch]
            arabic_texts = await asyncio.to_thread(translate_batch, english_texts)

            for block, arabic in zip(batch, arabic_texts):
                translated_blocks.append({
                    "english": block["text"],
                    "arabic": arabic,
                    "bbox": block["bbox"],
                })

            completed += len(batch)
            try:
                await status.edit_text(
                    "🌐 جاري ترجمة المحاضرة...\n\n"
                    f"📄 الصفحة: {page_index}/{total_pages}\n"
                    f"📝 الفقرات المترجمة: {completed}/{total_paragraphs}"
                )
            except Exception:
                logger.warning("Could not update progress message.")
        translated_pages.append(translated_blocks)
    return translated_pages


# ==========================================
# 5. تنسيق العربية وقياسها
# ==========================================

def prepare_arabic(text):
    text = text.strip()
    if not text:
        return ""
    return get_display(arabic_reshaper.reshape(text))


def wrap_arabic_by_width(text, font, fontsize, max_width):
    words = text.split()
    lines = []
    current = ""

    for word in words:
        candidate = f"{current} {word}".strip()
        shaped = prepare_arabic(candidate)
        try:
            candidate_width = font.text_length(shaped, fontsize=fontsize)
        except Exception:
            candidate_width = len(candidate) * fontsize * 0.65

        if candidate_width <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            # الكلمات الطويلة جداً تبقى كسطر واحد لتجنب إسقاطها
            current = word

    if current:
        lines.append(current)
    return lines or [""]


# ==========================================
# 6. إنشاء PDF: الترجمة الحمراء تحت كل فقرة
# ==========================================

def create_bilingual_pdf(source_path, translated_pages, translated_path):
    logger.info("Starting in-place bilingual PDF creation.")

    if not os.path.isfile(ARABIC_FONT):
        raise FileNotFoundError(
            f"خط العربية غير موجود: {ARABIC_FONT}\n"
            "ضع NotoNaskhArabic-Regular.ttf داخل مجلد البوت "
            "أو عيّن متغير ARABIC_FONT إلى مسار الخط."
        )

    output = fitz.open()
    original = fitz.open(source_path)

    try:
        if len(original) != len(translated_pages):
            raise ValueError("عدد الصفحات الأصلية لا يطابق عدد صفحات الترجمة.")

        font = fitz.Font(fontfile=ARABIC_FONT)

        for page_index, source_page in enumerate(original):
            # نسخ الصفحة الأصلية بنفس أبعادها، مع صورها ورسوماتها
            page = output.new_page(
                width=source_page.rect.width,
                height=source_page.rect.height,
            )
            page.show_pdf_page(page.rect, original, page_index)
            page.insert_font(fontname="arabic", fontfile=ARABIC_FONT)

            page_rect = page.rect
            blocks = translated_pages[page_index]

            # معالجة الفقرات من الأعلى إلى الأسفل حتى تبقى مرتبة
            blocks = sorted(
                blocks,
                key=lambda item: (item["bbox"][1], item["bbox"][0])
            )

            for block in blocks:
                arabic = (block.get("arabic") or "").strip()
                bbox = block.get("bbox")
                if not arabic or not bbox or len(bbox) != 4:
                    continue

                x0, y0, x1, y1 = map(float, bbox)
                # الترجمة تبدأ مباشرة بعد نهاية الفقرة الإنجليزية
                gap = 3.0
                left = max(page_rect.x0 + 6, x0)
                right = min(page_rect.x1 - 6, x1)
                top = y1 + gap

                if right <= left or top >= page_rect.y1 - 3:
                    logger.warning(
                        "No room below text block on page %s; skipping its Arabic.",
                        page_index + 1,
                    )
                    continue

                box_width = right - left
                available_height = page_rect.y1 - top - 3
                # حجم خط مبدئي صغير نسبياً حتى يناسب عرض الفقرة
                font_size = 9.5
                lines = wrap_arabic_by_width(arabic, font, font_size, box_width)

                # ارتفاع تقريبي لأسطر الترجمة، دون إنشاء مساحة أسفل الصفحة
                line_height = font_size * 1.55
                needed_height = max(line_height + 2, len(lines) * line_height + 2)
                box_height = min(needed_height, available_height)

                # إذا كان المكان ضيقاً، نقلل حجم الخط تدريجياً
                while box_height < needed_height and font_size > 6.0:
                    font_size -= 0.5
                    lines = wrap_arabic_by_width(arabic, font, font_size, box_width)
                    line_height = font_size * 1.55
                    needed_height = max(line_height + 2, len(lines) * line_height + 2)
                    box_height = min(needed_height, available_height)

                # منع الكتابة خارج الصفحة. قد لا تظهر ترجمة الفقرات القريبة جداً من أسفل الصفحة.
                rect = fitz.Rect(left, top, right, min(page_rect.y1 - 2, top + box_height))
                shaped_text = "\n".join(prepare_arabic(line) for line in lines)

                try:
                    page.insert_textbox(
                        rect,
                        shaped_text,
                        fontname="arabic",
                        fontsize=font_size,
                        color=(0.82, 0.0, 0.0),
                        align=fitz.TEXT_ALIGN_RIGHT,
                        lineheight=1.25,
                        overlay=True,
                    )
                except Exception as error:
                    logger.warning(
                        "Arabic insertion failed on page %s: %s",
                        page_index + 1, error,
                    )

            logger.info("Created page %s/%s.", page_index + 1, len(original))

        output.save(translated_path, garbage=4, deflate=True)

    finally:
        output.close()
        original.close()

    if not os.path.exists(translated_path) or os.path.getsize(translated_path) == 0:
        raise ValueError("لم يتم إنشاء ملف PDF صالح.")


# ==========================================
# 7. أوامر البوت
# ==========================================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "أهلاً بيك! 📚\n\n"
        "أرسل محاضرتك بصيغة PDF.\n"
        "راح أضيف الترجمة العربية بالأحمر تحت الفقرات الإنكليزية مباشرةً "
        "قدر الإمكان، مع الحفاظ على الصفحة الأصلية."
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📚 طريقة الاستخدام:\n\n"
        "1. أرسل ملف PDF.\n"
        "2. انتظر اكتمال الترجمة.\n"
        "3. استلم ملف المحاضرة المترجم."
    )


async def handle_pdf(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    document = message.document
    if not document:
        return

    filename = document.file_name or "lecture.pdf"
    if not filename.lower().endswith(".pdf"):
        await message.reply_text("❌ أرسل ملف PDF فقط.")
        return

    status = await message.reply_text("📥 استلمت المحاضرة!\nجاري تجهيز الملف...")

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = os.path.join(temp_dir, "source.pdf")
            translated_path = os.path.join(temp_dir, "bilingual_lecture.pdf")

            await status.edit_text("📥 جاري تنزيل ملف PDF...")
            tg_file = await context.bot.get_file(document.file_id)
            await tg_file.download_to_drive(source_path)

            with fitz.open(source_path) as pdf:
                page_count = len(pdf)
            if page_count == 0:
                raise ValueError("ملف PDF فارغ.")

            await status.edit_text(
                f"📖 عدد الصفحات: {page_count}\n"
                "🌐 جاري استخراج النصوص وترجمتها..."
            )

            translated_pages = await translate_document(source_path, status)

            await status.edit_text("📝 اكتملت الترجمة!\nجاري إنشاء ملف PDF...")
            await asyncio.to_thread(
                create_bilingual_pdf,
                source_path,
                translated_pages,
                translated_path,
            )

            await status.edit_text("✅ اكتملت المعالجة!\nجاري إرسال الملف...")
            with open(translated_path, "rb") as file:
                await context.bot.send_document(
                    chat_id=message.chat_id,
                    document=file,
                    filename="Bilingual_Lecture.pdf",
                    caption=(
                        "✅ تمت معالجة المحاضرة!\n"
                        "أُضيفت الترجمة الحمراء أسفل الفقرات الإنكليزية "
                        "قدر الإمكان مع الحفاظ على الصفحة الأصلية."
                    ),
                    connect_timeout=30,
                    read_timeout=180,
                    write_timeout=180,
                    pool_timeout=30,
                )
            await status.delete()

    except Exception as error:
        logger.exception("PDF processing failed: %s", error)
        error_text = str(error)[:700]
        try:
            await status.edit_text("❌ صار خطأ أثناء معالجة الملف:\n\n" + error_text)
        except Exception:
            await message.reply_text(f"❌ خطأ: {error_text}")


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.error("Unhandled error: %s", context.error, exc_info=context.error)


def main():
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_pdf))
    app.add_error_handler(error_handler)
    logger.info("Bot is running...")
    app.run_polling()


if __name__ == "__main__":
    main()
