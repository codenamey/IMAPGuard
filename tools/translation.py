from transformers import pipeline
from langdetect import detect, LangDetectException
from tqdm import tqdm

# Load translation models for multilingual analysis
marian_fi_to_en_model_name = "Helsinki-NLP/opus-mt-fi-en"
marian_en_to_fi_model_name = "Helsinki-NLP/opus-mt-en-fi"

translator_fi_to_en = pipeline("translation", model=marian_fi_to_en_model_name)
translator_en_to_fi = pipeline("translation", model=marian_en_to_fi_model_name)

def translate_email_content(content):
    try:
        detected_lang = detect(content)
        if detected_lang == "fi":
            return content  # No translation needed
        elif detected_lang == "en":
            return content  # No translation needed
        elif detected_lang != "fi":
            translation = translator_fi_to_en(content)[0]['translation_text']
            return translation
        else:
            translation = translator_en_to_fi(content)[0]['translation_text']
            return translation
    except LangDetectException as e:
        tqdm.write(f"Error detecting language: {e}")
        return content
