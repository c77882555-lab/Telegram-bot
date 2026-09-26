import os
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from google import genai
import fitz  # PyMuPDF
from docx import Document

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("مرحباً بك يا سجاد! أرسل ملف PDF وسأقوم بترجمته لك فوراً بدون أي قيود.")

def translate_with_gemini(text: str) -> str:
    if not text.strip():
        return ""
    try:
        if len(text) > 3000:
            text = text[:3000] + "..."
            
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=f"قم بترجمة النص الإنجليزي التالي إلى العربية الفصحى بدقة أكاديمية عالية:\n\n{text}"
        )
        return response.text.strip()
    except Exception as e:
        logger.error(f"Gemini Error: {e}")
        return "⚠️ (حدث ضغط مؤقت في الطلبات، جرب مرة أخرى)"

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    file = await message.document.get_file()
    file_name = message.document.file_name
    
    await message.reply_text("⏳ جاري قراءة الملف وترجمته عبر الذكاء الاصطناعي... انتظر ثوانٍ.")

    local_input = f"input_{file_name}"
    await file.download_to_drive(local_input)

    try:
        doc = fitz.open(local_input)
        full_text = ""
        for page in doc:
            t = page.get_text()
            if t:
                full_text += t + "\n"
        doc.close()

        if not full_text.strip():
            await message.reply_text("❌ لم أتمكن من العثور على نص داخل الملف. قد يكون صورة.")
            return

        translated_full = translate_with_gemini(full_text)

        output_docx = f"Translated_{file_name}.docx"
        out_doc = Document()
        out_doc.add_heading(f"ترجمة ملف: {file_name}", level=1)
        for paragraph in translated_full.split("\n"):
            if paragraph.strip():
                out_doc.add_paragraph(paragraph)
        out_doc.save(output_docx)

        await message.reply_document(
            document=open(output_docx, "rb"),
            filename=f"Translated_{file_name}.docx",
            caption="✅ **تمت ترجمة المحاضرة بنجاح وجاهزة للتحميل!** 📚"
        )

        if os.path.exists(output_docx):
            os.remove(output_docx)

    except Exception as e:
        logger.error(f"Error processing file: {e}")
        await message.reply_text("❌ حدث خطأ أثناء معالجة الملف.")
    
    finally:
        if os.path.exists(local_input):
            os.remove(local_input)

def main() -> None:
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    print("Bot is running...")
    application.run_polling()

if __name__ == "__main__":
    main()
