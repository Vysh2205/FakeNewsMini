import re
try:
    from langdetect import detect
except Exception:
    detect = None

try:
    from deep_translator import GoogleTranslator
except Exception:
    GoogleTranslator = None

SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "te": "Telugu",
    "es": "Spanish",
    "fr": "French",
    "de": "German"
}

def detect_and_translate(text: str):
    """
    Detects language of input text (supporting English, Hindi, Telugu, etc.).
    Translates to English if text is in Hindi/Telugu or another language.
    """
    if not text or not text.strip():
        return {
            "detected_language": "English",
            "language_code": "en",
            "translated_text": text,
            "was_translated": False
        }

    # Fallback Unicode script check for Hindi (Devanagari) & Telugu script
    is_hindi_script = bool(re.search(r'[\u0900-\u097F]', text))
    is_telugu_script = bool(re.search(r'[\u0C00-\u0C7F]', text))

    lang_code = "en"
    if is_hindi_script:
        lang_code = "hi"
    elif is_telugu_script:
        lang_code = "te"
    elif detect:
        try:
            detected = detect(text)
            if detected in SUPPORTED_LANGUAGES:
                lang_code = detected
        except Exception:
            lang_code = "en"

    lang_name = SUPPORTED_LANGUAGES.get(lang_code, lang_code.upper())
    translated_text = text
    was_translated = False

    if lang_code != "en":
        if GoogleTranslator:
            try:
                translated_text = GoogleTranslator(source='auto', target='en').translate(text)
                was_translated = True
            except Exception:
                translated_text = text
                was_translated = False

    return {
        "detected_language": lang_name,
        "language_code": lang_code,
        "translated_text": translated_text,
        "was_translated": was_translated
    }
