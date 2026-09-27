import os
import re
import asyncio
import tempfile
import logging
import time

import fitz
import arabic_reshaper
from bidi.algorithm import get_display
from deep_translator import GoogleTranslator
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
BATCH_SIZE = 4
MAX_RETRIES = 4

ARABIC_FONT = os.getenv("ARABIC_FONT", "NotoNaskhArabic-Regular.ttf")

if not BOT_TOKEN:
    raise RuntimeError("Missing TELEGRAM_TOKEN (or BOT_TOKEN)")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ==========================================
# 2. الترجمة عبر Google Translate (deep-translator)
# ==========================================

def translate_text(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""

    # Google Translate through the unofficial deep-translator library.
    # No API key is required, but Google may throttle or block frequent requests.
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            result = GoogleTranslator(source="auto", target="ar").translate(text)
            if not result or not result.strip():
                raise ValueError("Google Translate returned an empty translation.")
            return result.strip()
        except Exception as error:
            last_error = error
            wait_time = min(2 ** attempt, 12)
            logger.warning(
                "Google Translate failed (attempt %s/%s): %s",
                attempt + 1, MAX_RETRIES, error,
            )
            if attempt < MAX_RETRIES - 1:
                time.sleep(wait_time)
    raise RuntimeError(f"Google Translate failed after retries: {last_error}")


def translate_batch(texts):
    # Translate each paragraph separately to preserve exact paragraph alignment.
    return [translate_text(text) for text in texts]


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
# 6. إنشاء PDF: إدراج الترجمة تحت كل فقرة
# ==========================================

def create_bilingual_pdf(source_path, translated_pages, translated_path):
    """
    يضيف الترجمة الحمراء مباشرة بعد كل كتلة نصية.
    تُقسّم الصفحة الأصلية إلى مقاطع أفقية، ويُدرج بين المقاطع
    شريط للترجمة، ثم يُزاح ما تبقى من الصفحة إلى الأسفل.
    هذا يمنع تداخل الترجمة مع الفقرات التالية ويحافظ على الرسومات.
    """
    if not os.path.isfile(ARABIC_FONT):
        raise FileNotFoundError(
            f"خط العربية غير موجود: {ARABIC_FONT}\n"
            "ضع NotoNaskhArabic-Regular.ttf داخل مجلد البوت "
            "أو عيّن متغير ARABIC_FONT إلى مسار الخط."
        )

    original = fitz.open(source_path)
    output = fitz.open()

    try:
        if len(original) != len(translated_pages):
            raise ValueError("عدد الصفحات الأصلية لا يطابق عدد صفحات الترجمة.")

        font = fitz.Font(fontfile=ARABIC_FONT)
        font_size = 18.0
        margin_x = 28
        gap_before_translation = 3
        gap_after_translation = 8
        line_height = font_size * 1.9

        def wrap_arabic(text, max_width):
            words = (text or "").split()
            lines = []
            current = ""
            for word in words:
                candidate = f"{current} {word}".strip()
                shaped = prepare_arabic(candidate)
                try:
                    width = font.text_length(shaped, fontsize=font_size)
                except Exception:
                    width = len(candidate) * font_size * 0.65

                if width <= max_width:
                    current = candidate
                else:
                    if current:
                        lines.append(current)
                    current = word
            if current:
                lines.append(current)
            return lines or [""]

        for page_index, source_page in enumerate(original):
            page_rect = source_page.rect
            page_width = page_rect.width
            page_height = page_rect.height
            page_blocks = translated_pages[page_index]

            # ترتيب الفقرات לפי نهاية الفقرة حتى نضيف الترجمة تحتها
            candidates = []
            for item in page_blocks:
                arabic = (item.get("arabic") or "").strip()
                bbox = item.get("bbox")
                if not arabic or not bbox or len(bbox) != 4:
                    continue
                x0, y0, x1, y1 = map(float, bbox)
                y1 = min(max(y1, 0), page_height)
                if y1 <= 0:
                    continue
                lines = wrap_arabic(arabic, page_width - 2 * margin_x)
                text_height = max(1, len(lines)) * line_height + 6
                candidates.append({
                    "y0": max(0, float(y0)),
                    "y1": y1,
                    "arabic": arabic,
                    "lines": lines,
                    "height": text_height,
                })

            candidates.sort(key=lambda item: (item["y1"], item["y0"]))

            # اجمع الكتل التي تنتهي تقريباً عند نفس المستوى لتجنب تقسيم الصفحة مرتين
            groups = []
            for item in candidates:
                if groups and abs(item["y1"] - groups[-1]["y1"]) <= 3:
                    groups[-1]["items"].append(item)
                    groups[-1]["y1"] = max(groups[-1]["y1"], item["y1"])
                else:
                    groups.append({"y1": item["y1"], "items": [item]})

            # لا نترجم النصوص القصيرة جداً التي تكون غالباً أرقام صفحات أو تسميات صغيرة
            # ونبقي الفقرات والقوائم والعناوين النصية.
            filtered_groups = []
            for group in groups:
                valid_items = [
                    item for item in group["items"]
                    if len(item["arabic"]) >= 8
                ]
                if valid_items:
                    group["items"] = valid_items
                    filtered_groups.append(group)

            # احسب ارتفاع كل شريط ترجمة قبل إنشاء الصفحة
            for group in filtered_groups:
                group["strip_height"] = (
                    gap_before_translation
                    + sum(item["height"] for item in group["items"])
                    + gap_after_translation
                )

            total_extra = sum(group["strip_height"] for group in filtered_groups)
            new_page = output.new_page(
                width=page_width,
                height=page_height + total_extra,
            )
            new_page.insert_font(fontname="arabic", fontfile=ARABIC_FONT)

            source_cursor = 0.0
            target_cursor = 0.0

            for group in filtered_groups:
                cut_y = min(max(group["y1"], source_cursor), page_height)

                # انسخ المقطع الأصلي كما هو، مع إزاحة المقطع اللاحق للأسفل
                if cut_y > source_cursor:
                    source_clip = fitz.Rect(
                        0, source_cursor, page_width, cut_y
                    )
                    target_rect = fitz.Rect(
                        0, target_cursor, page_width,
                        target_cursor + (cut_y - source_cursor),
                    )
                    new_page.show_pdf_page(
                        target_rect,
                        original,
                        page_index,
                        clip=source_clip,
                        overlay=True,
                    )
                    target_cursor += cut_y - source_cursor
                    source_cursor = cut_y

                target_cursor += gap_before_translation

                for item in group["items"]:
                    block_height = item["height"]
                    translation_rect = fitz.Rect(
                        margin_x,
                        target_cursor,
                        page_width - margin_x,
                        target_cursor + block_height,
                    )
                    shaped = "\n".join(
                        prepare_arabic(line) for line in item["lines"]
                    )
                    new_page.insert_textbox(
                        translation_rect,
                        shaped,
                        fontname="arabic",
                        fontsize=font_size,
                        color=(0.82, 0.0, 0.0),
                        align=fitz.TEXT_ALIGN_RIGHT,
                        lineheight=1.3,
                        overlay=True,
                    )
                    target_cursor += block_height

                target_cursor += gap_after_translation

            # انسخ ما تبقى من الصفحة الأصلية بعد آخر ترجمة
            if source_cursor < page_height:
                source_clip = fitz.Rect(
                    0, source_cursor, page_width, page_height
                )
                target_rect = fitz.Rect(
                    0, target_cursor, page_width,
                    target_cursor + (page_height - source_cursor),
                )
                new_page.show_pdf_page(
                    target_rect,
                    original,
                    page_index,
                    clip=source_clip,
                    overlay=True,
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
        user_error = (
            "❌ صار خطأ أثناء الترجمة أو معالجة الملف. ممكن Google Translate "
            "قيّد الطلبات مؤقتاً؛ انتظر شوي وجرب مرة ثانية.\n\n" + error_text
        )
        try:
            await status.edit_text(user_error)
        except Exception:
            await message.reply_text(user_error)


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
