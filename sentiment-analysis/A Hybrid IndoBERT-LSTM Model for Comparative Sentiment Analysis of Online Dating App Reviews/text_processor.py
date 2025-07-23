import pandas as pd
import re
from bs4 import BeautifulSoup

def _map_sentiment(score: int) -> int:
    """Helper function untuk mengubah skor rating menjadi label biner."""
    return 1 if score >= 4 else 0 # 1 untuk Positif, 0 untuk Negatif

def _clean_text_for_bert(text: str) -> str:
    """Helper function untuk pembersihan teks minimalis untuk BERT."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'http\S+|www\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s]', '', text) # Hapus emoji dan simbol
    return text

def process_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """
    Memproses DataFrame hasil crawling menjadi format yang siap untuk model.

    Args:
        df (pd.DataFrame): DataFrame dari fungsi crawl_reviews.

    Returns:
        pd.DataFrame: DataFrame yang bersih dengan kolom 'review' dan 'label'.
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