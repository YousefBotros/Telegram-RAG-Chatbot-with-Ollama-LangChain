"""
Telegram Bot with RAG capabilities
"""

import asyncio
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
    ContextTypes
)
from config import config
from rag_chain import rag_chain
from database import db
import os

# Store user chat history (in production, use Redis or database)
user_histories = {}

def get_user_history(user_id):
    """Get or create user chat history"""
    if user_id not in user_histories:
        user_histories[user_id] = []
    return user_histories[user_id]

def add_to_history(user_id, role, content):
    """Add message to user history"""
    history = get_user_history(user_id)
    history.append({"role": role, "content": content})
    
    # Keep only last N messages
    if len(history) > config.MAX_HISTORY * 2:
        user_histories[user_id] = history[-config.MAX_HISTORY * 2:]

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command"""
    user = update.effective_user
    
    welcome_message = f"""
🌟 Welcome to MedCare AI Assistant, {user.first_name}! 🌟

I'm an AI-powered healthcare assistant that can:
✅ Answer medical questions (based on my knowledge base)
✅ Provide health information and guidance
✅ Remember our conversation context
✅ Support both English and Arabic

🔍 *How to use me:*
• Just type your question naturally
• Upload PDF/TXT files to add to my knowledge base
• Use /help to see all commands

⚕️ *Important:* I provide informational support only. Always consult a doctor for medical advice.

Let's get started! Ask me anything about health, medicine, or wellness. 💙
"""
    
    await update.message.reply_text(welcome_message, parse_mode="Markdown")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help command"""
    help_text = """
📚 *Available Commands*

/start - Start the bot
/help - Show this help message
/clear - Clear conversation history
/status - Show bot status
/about - About this bot
/feedback - Send feedback

💡 *How to use:*
Simply type your question in English or Arabic!

📎 *File Support:*
Upload PDF or TXT files to add them to my knowledge base

🔄 *Memory:* I remember our conversation context!
"""
    await update.message.reply_text(help_text, parse_mode="Markdown")

async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Clear user conversation history"""
    user_id = update.effective_user.id
    if user_id in user_histories:
        user_histories[user_id] = []
    await update.message.reply_text("✅ Conversation history cleared!")

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show bot status"""
    user_id = update.effective_user.id
    history_length = len(get_user_history(user_id))
    
    # Check Ollama status
    import requests
    ollama_status = "❌ Not connected"
    try:
        response = requests.get(f"{config.OLLAMA_HOST}/api/tags", timeout=5)
        if response.status_code == 200:
            ollama_status = "✅ Connected"
    except:
        pass
    
    status_text = f"""
🤖 *Bot Status*

Ollama: {ollama_status}
Model: {config.MODEL_NAME}
Your history length: {history_length} messages
RAG enabled: ✅
"""
    await update.message.reply_text(status_text, parse_mode="Markdown")

async def about_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show about info"""
    about_text = """
🧠 *MedCare AI Assistant*

Version: 1.0
Technology:
• LangChain RAG Pipeline
• Ollama LLM (Llama 3.1)
• ChromaDB Vector Store
• Sentence Transformers
• Python Telegram Bot

Created by: Yousef Botros
Purpose: AI-powered healthcare information assistant

*Note:* For medical emergencies, please contact your doctor immediately.
"""
    await update.message.reply_text(about_text, parse_mode="Markdown")

async def handle_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle uploaded files (PDF/TXT)"""
    user = update.effective_user
    document = update.message.document
    
    file_name = document.file_name
    file_extension = file_name.split('.')[-1].lower()
    
    if file_extension not in ['pdf', 'txt']:
        await update.message.reply_text("❌ Please upload PDF or TXT files only.")
        return
    
    await update.message.reply_text(f"📥 Received file: {file_name}\nProcessing...")
    
    # Download file
    file = await document.get_file()
    
    # Save to knowledge base directory
    os.makedirs(config.KNOWLEDGE_BASE_DIR, exist_ok=True)
    file_path = os.path.join(config.KNOWLEDGE_BASE_DIR, file_name)
    
    await file.download_to_drive(file_path)
    
    # Add to vector database
    try:
        num_chunks = db.add_document(file_path)
        await update.message.reply_text(
            f"✅ Document added successfully!\n"
            f"📄 {file_name}\n"
            f"📊 Created {num_chunks} chunks\n"
            f"🔍 Now you can ask questions about this document!"
        )
    except Exception as e:
        await update.message.reply_text(f"❌ Error adding document: {str(e)}")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle text messages"""
    user_id = update.effective_user.id
    user_message = update.message.text
    
    # Show typing indicator
    await update.message.chat.send_action(action="typing")
    
    # Get user history
    history = get_user_history(user_id)
    
    # Add user message to history
    add_to_history(user_id, "user", user_message)
    
    # Get response from RAG chain
    response = rag_chain.get_response(user_message, history)
    
    # Add assistant response to history
    add_to_history(user_id, "assistant", response)
    
    # Split long messages
    if len(response) > 4000:
        parts = [response[i:i+4000] for i in range(0, len(response), 4000)]
        for part in parts:
            await update.message.reply_text(part)
    else:
        await update.message.reply_text(response)

async def feedback_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Collect user feedback"""
    await update.message.reply_text(
        "📝 Please send your feedback about the bot.\n"
        "Include what you liked, what can be improved, or feature requests.\n\n"
        "Type /cancel to exit feedback mode."
    )
    context.user_data['feedback_mode'] = True

async def handle_feedback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle feedback messages"""
    if context.user_data.get('feedback_mode'):
        feedback = update.message.text
        user = update.effective_user
        
        # Save feedback to file (in production, save to database)
        with open("feedback.txt", "a", encoding="utf-8") as f:
            f.write(f"User: {user.id} ({user.username})\n")
            f.write(f"Feedback: {feedback}\n")
            f.write("-" * 50 + "\n")
        
        await update.message.reply_text("✅ Thank you for your feedback! It helps us improve.")
        context.user_data['feedback_mode'] = False
        return True
    return False

def main():
    """Main function to run the bot"""
    
    # Create application
    application = Application.builder().token(config.TELEGRAM_TOKEN).build()
    
    # Add command handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("clear", clear_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("about", about_command))
    application.add_handler(CommandHandler("feedback", feedback_command))
    
    # Add message handlers
    application.add_handler(MessageHandler(filters.Document.ALL, handle_file))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    print("🤖 Bot is running...")
    print(f"Bot username: @{application.bot.username}" if application.bot.username else "Bot started")
    print("Press Ctrl+C to stop")
    
    # Start the bot
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
