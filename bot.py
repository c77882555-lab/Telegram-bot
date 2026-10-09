import os
import asyncio
import tempfile
import logging
import time

import fitz
import requests
import arabic_reshaper
from bidi.algorithm import get_display

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# ==========================================
# 1. الإعدادات
# ==========================================

BOT_TOKEN = (
    os.getenv("TELEGRAM_TOKEN")
    or os.getenv("BOT_TOKEN")
    or os.getenv("TELEGRAM_BOT_TOKEN")
)

DEEPL_AUTH_KEY = os.getenv("DEEPL_AUTH_KEY")

ARABIC_FONT = os.getenv(
    "ARABIC_FONT",
    "NotoNaskhArabic-Regular.ttf",
)

BATCH_SIZE = 4
FONT_SIZE = 12.0

DEEPL_URL = "https://api-free.deepl.com/v2/translate"

if not BOT_TOKEN:
    raise RuntimeError("Missing Telegram bot token.")

if not DEEPL_AUTH_KEY:
    raise RuntimeError("Missing DEEPL_AUTH_KEY environment variable.")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ==========================================
# 2. DeepL API
# ==========================================

def translate_batch(texts):
    texts = [text.strip() for text in texts if text and text.strip()]

    if not texts:
        return []

    last_error = None

    for attempt in range(3):
        try:
            # إرسال عدة فقرات بطلب واحد لتقليل عدد الطلبات
            payload = [
                ("text", text)
                for text in texts
            ]
            payload.extend([
                ("source_lang", "EN"),
                ("target_lang", "AR"),
            ])

            response = requests.post(
                DEEPL_URL,
                headers={
                    "Authorization": f"DeepL-Auth-Key {DEEPL_AUTH_KEY}",
                },
                data=payload,
                timeout=60,
            )

            if response.status_code == 429:
                raise RuntimeError(
                    "DeepL rate limit reached. Try again later."
                )

            response.raise_for_status()
            data = response.json()

            translations = data.get("translations", [])

            if len(translations) != len(texts):
                raise RuntimeError(
                    "DeepL returned an unexpected number of translations."
                )

            return [
                item["text"].strip()
                for item in translations
            ]

        except Exception as error:
            last_error = error
            logger.warning(
                "DeepL attempt %s failed: %s",
                attempt + 1,
                error,
            )

            if attempt < 2:
                time.sleep(2 ** (attempt + 1))

    raise RuntimeError(
        f"DeepL translation failed: {last_error}"
    )


def translate_text(text):
    text = (text or "").strip()

    if not text:
        return ""

    return translate_batch([text])[0]


# ==========================================
# 3. استخراج النصوص من PDF
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

        all_blocks = [
            extract_blocks(page)
            for page in pdf
        ]

    total_blocks = sum(len(blocks) for blocks in all_blocks)

    if total_blocks == 0:
        raise ValueError(
            "لم يتم العثور على نص قابل للاستخراج. "
            "قد يكون الملف ممسوحاً ضوئياً ويحتاج OCR."
        )

    completed = 0

    for page_number, blocks in enumerate(all_blocks, start=1):
        page_translations = []

        for start in range(0, len(blocks), BATCH_SIZE):
            batch = blocks[start:start + BATCH_SIZE]

            english_texts = [
                block["text"]
                for block in batch
            ]

            arabic_texts = await asyncio.to_thread(
                translate_batch,
                english_texts,
            )

            for block, arabic in zip(batch, arabic_texts):
                page_translations.append({
                    "english": block["text"],
                    "arabic": arabic,
                    "bbox": block["bbox"],
                })

            completed += len(batch)

            try:
                await status.edit_text(
                    "🌐 جاري ترجمة المحاضرة...\n\n"
                    f"📄 الصفحة: {page_number}/{total_pages}\n"
                    f"📝 الفقرات: {completed}/{total_blocks}"
                )
            except Exception:
                logger.warning("Could not update progress message.")

        translated_pages.append(page_translations)

    return translated_pages


# ==========================================
# 4. تجهيز النص العربي
# ==========================================

def prepare_arabic(text):
    text = (text or "").strip()

    if not text:
        return ""

    return get_display(
        arabic_reshaper.reshape(text)
    )


def wrap_arabic(text, font, fontsize, max_width):
    words = (text or "").split()

    if not words:
        return [""]

    lines = []
    current = ""

    for word in words:
        candidate = f"{current} {word}".strip()
        shaped = prepare_arabic(candidate)

        try:
            width = font.text_length(
                shaped,
                fontsize=fontsize,
            )
        except Exception:
            width = len(candidate) * fontsize * 0.65

        if width <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)

            current = word

    if current:
        lines.append(current)

    return lines or [""]


# ==========================================
# 5. إنشاء PDF ثنائي اللغة
# ==========================================

def create_bilingual_pdf(
    source_path,
    translated_pages,
    translated_path,
):
    if not os.path.isfile(ARABIC_FONT):
        raise FileNotFoundError(
            f"خط العربية غير موجود: {ARABIC_FONT}. "
            "ضع ملف NotoNaskhArabic-Regular.ttf "
            "داخل مجلد البوت."
        )

    original = fitz.open(source_path)
    output = fitz.open()

    try:
        if len(original) != len(translated_pages):
            raise ValueError(
                "عدد الصفحات الأصلية لا يطابق صفحات الترجمة."
            )

        margin_x = 28
        gap_before = 3
        gap_after = 8
        line_height = FONT_SIZE * 1.8

        font = fitz.Font(fontfile=ARABIC_FONT)

        for page_index, source_page in enumerate(original):
            page_width = source_page.rect.width
            page_height = source_page.rect.height
            blocks = translated_pages[page_index]

            candidates = []

            for item in blocks:
                arabic = (item.get("arabic") or "").strip()
                bbox = item.get("bbox")

                if not arabic or not bbox or len(bbox) != 4:
                    continue

                x0, y0, x1, y1 = map(float, bbox)
                y1 = min(max(y1, 0), page_height)

                if y1 <= 0:
                    continue

                lines = wrap_arabic(
                    arabic,
                    font,
                    FONT_SIZE,
                    page_width - 2 * margin_x,
                )

                height = max(1, len(lines)) * line_height + 6

                candidates.append({
                    "y0": max(0, y0),
                    "y1": y1,
                    "lines": lines,
                    "arabic": arabic,
                    "height": height,
                })

            candidates.sort(
                key=lambda item: (item["y1"], item["y0"])
            )

            groups = []

            for item in candidates:
                if (
                    groups
                    and abs(item["y1"] - groups[-1]["y1"]) <= 3
                ):
                    groups[-1]["items"].append(item)
                    groups[-1]["y1"] = max(
                        groups[-1]["y1"],
                        item["y1"],
                    )
                else:
                    groups.append({
                        "y1": item["y1"],
                        "items": [item],
                    })

            filtered_groups = []

            for group in groups:
                valid = [
                    item
                    for item in group["items"]
                    if len(item["arabic"]) >= 2
                ]

                if valid:
                    group["items"] = valid
                    group["strip_height"] = (
                        gap_before
                        + sum(item["height"] for item in valid)
                        + gap_after
                    )
                    filtered_groups.append(group)

            extra_height = sum(
                group["strip_height"]
                for group in filtered_groups
            )

            new_page = output.new_page(
                width=page_width,
                height=page_height + extra_height,
            )

            new_page.insert_font(
                fontname="arabic",
                fontfile=ARABIC_FONT,
            )

            source_cursor = 0.0
            target_cursor = 0.0

            for group in filtered_groups:
                cut_y = min(
                    max(group["y1"], source_cursor),
                    page_height,
                )

                if cut_y > source_cursor:
                    source_clip = fitz.Rect(
                        0,
                        source_cursor,
                        page_width,
                        cut_y,
                    )

                    target_rect = fitz.Rect(
                        0,
                        target_cursor,
                        page_width,
                        target_cursor + cut_y - source_cursor,
                    )

                    new_page.show_pdf_page(
                        target_rect,
                        original,
                        page_index,
                        clip=source_clip,
                    )

                    target_cursor += cut_y - source_cursor
                    source_cursor = cut_y

                target_cursor += gap_before

                for item in group["items"]:
                    height = item["height"]

                    rect = fitz.Rect(
                        margin_x,
                        target_cursor,
                        page_width - margin_x,
                        target_cursor + height,
                    )

                    shaped = "\n".join(
                        prepare_arabic(line)
                        for line in item["lines"]
                    )

                    new_page.insert_textbox(
                        rect,
                        shaped,
                        fontname="arabic",
                        fontsize=FONT_SIZE,
                        color=(0.82, 0.0, 0.0),
                        align=fitz.TEXT_ALIGN_RIGHT,
                        lineheight=1.3,
                    )

                    target_cursor += height

                target_cursor += gap_after

            if source_cursor < page_height:
                source_clip = fitz.Rect(
                    0,
                    source_cursor,
                    page_width,
                    page_height,
                )

                target_rect = fitz.Rect(
                    0,
                    target_cursor,
                    page_width,
                    target_cursor + page_height - source_cursor,
                )

                new_page.show_pdf_page(
                    target_rect,
                    original,
                    page_index,
                    clip=source_clip,
                )

            logger.info(
                "Created page %s/%s",
                page_index + 1,
                len(original),
            )

        output.save(
            translated_path,
            garbage=4,
            deflate=True,
        )

    finally:
        output.close()
        original.close()

    if (
        not os.path.isfile(translated_path)
        or os.path.getsize(translated_path) == 0
    ):
        raise ValueError("لم يتم إنشاء ملف PDF صالح.")


# ==========================================
# 6. أوامر البوت
# ==========================================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "أهلاً بيك! 📚\n\n"
        "أرسل محاضرتك بصيغة PDF.\n"
        "راح أضيف الترجمة العربية باللون الأحمر "
        "تحت الفقرات الإنكليزية قدر الإمكان."
    )


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
        await message.reply_text("❌ أرسل ملف PDF فقط.")
        return

    status = await message.reply_text(
        "📥 استلمت المحاضرة!\nجاري تجهيز الملف..."
    )

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = os.path.join(temp_dir, "source.pdf")
            translated_path = os.path.join(
                temp_dir,
                "bilingual_lecture.pdf",
            )

            await status.edit_text("📥 جاري تنزيل ملف PDF...")

            tg_file = await context.bot.get_file(document.file_id)
            await tg_file.download_to_drive(source_path)

            with fitz.open(source_path) as pdf:
                page_count = len(pdf)

            if page_count == 0:
                raise ValueError("ملف PDF فارغ.")

            await status.edit_text(
                f"📖 عدد الصفحات: {page_count}\n"
                "🌐 جاري ترجمة المحاضرة..."
            )

            translated_pages = await translate_document(
                source_path,
                status,
            )

            await status.edit_text(
                "📝 اكتملت الترجمة!\nجاري إنشاء PDF..."
            )

            await asyncio.to_thread(
                create_bilingual_pdf,
                source_path,
                translated_pages,
                translated_path,
            )

            await status.edit_text("📤 جاري إرسال الملف...")

            with open(translated_path, "rb") as file:
                await context.bot.send_document(
                    chat_id=message.chat_id,
                    document=file,
                    filename="Bilingual_Lecture.pdf",
                    caption=(
                        "✅ تمت معالجة المحاضرة!\n"
                        "الترجمة العربية باللون الأحمر "
                        "تحت الفقرات الإنجليزية قدر الإمكان."
                    ),
                    connect_timeout=30,
                    read_timeout=180,
                    write_timeout=180,
                    pool_timeout=30,
                )

            await status.delete()

    except Exception as error:
        logger.exception("PDF processing failed: %s", error)

        user_error = (
            "❌ حدث خطأ أثناء معالجة الملف.\n\n"
            + str(error)[:700]
        )

        try:
            await status.edit_text(user_error)
        except Exception:
            await message.reply_text(user_error)


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
# 7. تشغيل البوت
# ==========================================

def main():
    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))

    app.add_handler(
        MessageHandler(filters.Document.ALL, handle_pdf)
    )

    app.add_error_handler(error_handler)

    logger.info("Bot is running...")
    app.run_polling()


if __name__ == "__main__":
    main()
