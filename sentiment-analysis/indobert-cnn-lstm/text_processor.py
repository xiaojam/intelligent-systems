import pandas as pd
import re
from bs4 import BeautifulSoup

def _map_sentiment(score: int) -> int:
    """
    Mengubah skor rating menjadi label biner (1: Positif, 0: Negatif).
    """
    return 1 if score >= 4 else 0

def _clean_text_for_bert(text: str) -> str:
    """
    Pembersihan teks minimalis yang cocok untuk model BERT.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'http\S+|www\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s]', '', text)
    return text

def process_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """
    Memproses DataFrame mentah menjadi format siap pakai untuk model.
    """
    print("Memulai pemrosesan data...")
    df_processed = df[['content', 'score']].copy()
    df_processed.rename(columns={'content': 'review'}, inplace=True)
    df_processed.dropna(subset=['review'], inplace=True)
    df_processed['label'] = df_processed['score'].apply(_map_sentiment)
    df_processed['cleaned_review'] = df_processed['review'].apply(_clean_text_for_bert)
    final_df = df_processed[['review', 'cleaned_review', 'score', 'label']].copy()
    print("Pemrosesan data selesai.")
    return final_df