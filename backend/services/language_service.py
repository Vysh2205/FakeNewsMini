import re
import requests
import urllib.parse

try:
    from langdetect import detect
except Exception:
    detect = None

SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "te": "Telugu",
    "ta": "Tamil",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mr": "Marathi",
    "bn": "Bengali",
    "es": "Spanish",
    "fr": "French",
    "de": "German"
}

def translate_to_english(text: str, lang_code: str = "auto") -> str:
    """
    Multi-stage robust translation to English.
    Tier 1: Google Chrome Translate Client5 API (High Reliability)
    Tier 2: MyMemory Translation API
    """
    if not text or not text.strip():
        return text

    # Tier 1: Google Chrome Translate Client5 API
    try:
        url = f"https://clients5.google.com/translate_a/t?client=dict-chrome-ex&sl={lang_code}&tl=en&q={urllib.parse.quote(text)}"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        resp = requests.get(url, headers=headers, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list) and len(data) > 0:
                if isinstance(data[0], list) and len(data[0]) > 0:
                    translated = data[0][0]
                    if translated and isinstance(translated, str) and len(translated.strip()) > 0:
                        return translated
                elif isinstance(data[0], str):
                    return data[0]
    except Exception:
        pass

    # Tier 2: MyMemory Translation API
    try:
        pair = f"{lang_code}|en" if lang_code != "auto" else "te|en"
        url = f"https://api.mymemory.translated.net/get?q={urllib.parse.quote(text)}&langpair={pair}"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            res_data = resp.json()
            translated = res_data.get("responseData", {}).get("translatedText", "")
            if translated and len(translated.strip()) > 0 and not translated.startswith("MYMEMORY WARNING"):
                return translated
    except Exception:
        pass

    return text

def detect_and_translate(text: str):
    """
    Detects language of input text (supporting English, Hindi, Telugu, Tamil, etc.).
    Translates to English if text is in Hindi/Telugu or another language.
    """
    if not text or not text.strip():
        return {
            "detected_language": "English",
            "language_code": "en",
            "translated_text": text,
            "was_translated": False
        }

    # Script regex pattern detection
    is_hindi_script = bool(re.search(r'[\u0900-\u097F]', text))
    is_telugu_script = bool(re.search(r'[\u0C00-\u0C7F]', text))
    is_tamil_script = bool(re.search(r'[\u0B80-\u0BFF]', text))
    is_kannada_script = bool(re.search(r'[\u0C80-\u0CFF]', text))
    is_malayalam_script = bool(re.search(r'[\u0D00-\u0D7F]', text))
    is_bengali_script = bool(re.search(r'[\u0980-\u09FF]', text))

    lang_code = "en"
    if is_hindi_script:
        lang_code = "hi"
    elif is_telugu_script:
        lang_code = "te"
    elif is_tamil_script:
        lang_code = "ta"
    elif is_kannada_script:
        lang_code = "kn"
    elif is_malayalam_script:
        lang_code = "ml"
    elif is_bengali_script:
        lang_code = "bn"
    elif detect:
        try:
            detected = detect(text)
            if detected in SUPPORTED_LANGUAGES:
                lang_code = detected
            elif detected != "en":
                lang_code = detected
        except Exception:
            lang_code = "en"

    lang_name = SUPPORTED_LANGUAGES.get(lang_code, lang_code.upper())
    translated_text = text
    was_translated = False

    if lang_code != "en" or is_hindi_script or is_telugu_script or is_tamil_script or is_kannada_script or is_malayalam_script or is_bengali_script:
        translated_result = translate_to_english(text, lang_code)
        if translated_result and translated_result != text:
            translated_text = translated_result
            was_translated = True

    return {
        "detected_language": lang_name,
        "language_code": lang_code,
        "translated_text": translated_text,
        "was_translated": was_translated
    }
