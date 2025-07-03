import re
import nltk
from nltk.corpus import stopwords

def initialize_stopwords():
    try:
        return set(stopwords.words('indonesian'))
    except LookupError:
        print("Mengunduh stopwords Bahasa Indonesia untuk pertama kali...")
        nltk.download('stopwords')
        return set(stopwords.words('indonesian'))

indonesian_stopwords = initialize_stopwords()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_stopwords(text):
    tokens = text.split()
    filtered_tokens = [word for word in tokens if word not in indonesian_stopwords]
    return ' '.join(filtered_tokens)

def preprocess(text):
    return remove_stopwords(clean_text(text))