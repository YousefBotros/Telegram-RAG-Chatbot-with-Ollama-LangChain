"""
Utility functions for the Telegram bot
"""

import re
from typing import List, Dict

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters (keep basic punctuation)
    text = re.sub(r'[^\w\s\u0600-\u06FF.,!?]', '', text)
    return text.strip()

def format_response(text: str) -> str:
    """Format response for Telegram"""
    # Escape Markdown special characters
    special_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
    for char in special_chars:
        text = text.replace(char, f'\\{char}')
    return text

def is_medical_query(text: str) -> bool:
    """Check if query is medical-related"""
    medical_keywords = [
        'symptom', 'pain', 'fever', 'cough', 'headache', 'medicine', 'drug',
        'treatment', 'doctor', 'hospital', 'health', 'disease', 'diagnosis',
        'prescription', 'allergy', 'infection', 'surgery', 'therapy'
    ]
    
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in medical_keywords)

def get_response_style(language: str) -> Dict:
    """Get response style based on language"""
    styles = {
        'en': {
            'prefix': '',
            'suffix': '',
            'disclaimer': '\n\n⚠️ *Disclaimer*: This is AI-generated information. Please consult a healthcare professional.'
        },
        'ar': {
            'prefix': '',
            'suffix': '',
            'disclaimer': '\n\n⚠️ *تنويه*: هذه معلومات تم إنشاؤها بواسطة الذكاء الاصطناعي. يرجى استشارة أخصائي الرعاية الصحية.'
        }
    }
    return styles.get(language, styles['en'])

def detect_language(text: str) -> str:
    """Detect if text is Arabic or English"""
    arabic_pattern = re.compile(r'[\u0600-\u06FF]')
    if arabic_pattern.search(text):
        return 'ar'
    return 'en'

def chunk_text(text: str, max_length: int = 4000) -> List[str]:
    """Split long text into chunks"""
    if len(text) <= max_length:
        return [text]
    
    chunks = []
    words = text.split()
    current_chunk = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) + 1 <= max_length:
            current_chunk.append(word)
            current_length += len(word) + 1
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = len(word) + 1
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks
