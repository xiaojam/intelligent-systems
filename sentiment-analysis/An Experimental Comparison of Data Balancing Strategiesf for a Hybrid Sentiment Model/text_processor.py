# text_processor.py

import pandas as pd
import re
from bs4 import BeautifulSoup

def _map_sentiment(score: int) -> int:
    return 1 if score >= 4 else 0

def _clean_text_for_bert(text: str) -> str:
    if not isinstance(text, str): return ""
    text = text.lower()
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'http\S+|www\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s]', '', text)
    return text

def _balance_with_undersampling(df: pd.DataFrame) -> pd.DataFrame:
    """Hanya melakukan undersampling pada kelas mayoritas."""
    print("Menjalankan strategi: Undersampling Murni...")
    class_counts = df['label'].value_counts()
    min_class_count = class_counts.min()
    max_class_name = class_counts.idxmax()
    df_majority = df[df['label'] == max_class_name]
    df_minority = df[df['label'] != max_class_name]
    df_majority_undersampled = df_majority.sample(n=min_class_count, random_state=42)
    return pd.concat([df_majority_undersampled, df_minority]).sample(frac=1, random_state=42).reset_index(drop=True)

def _balance_with_oversampling(df: pd.DataFrame) -> pd.DataFrame:
    """Hanya melakukan oversampling pada kelas minoritas."""
    print("Menjalankan strategi: Oversampling Murni...")
    class_counts = df['label'].value_counts()
    max_class_count = class_counts.max()
    minority_class_name = class_counts.idxmin()
    df_majority = df[df['label'] != minority_class_name]
    df_minority = df[df['label'] == minority_class_name]
    df_minority_oversampled = df_minority.sample(n=max_class_count, replace=True, random_state=42)
    return pd.concat([df_majority, df_minority_oversampled]).sample(frac=1, random_state=42).reset_index(drop=True)

def _hybrid_balance_data(df: pd.DataFrame, balance_ratio: float) -> pd.DataFrame:
    """Melakukan oversampling jika perlu, lalu undersampling."""
    print(f"Menjalankan strategi: Hybrid Sampling (Target Rasio: {balance_ratio})...")
    total_samples = len(df)
    target_samples_per_class = int(total_samples * balance_ratio)
    class_counts = df['label'].value_counts()
    minority_count = class_counts.min()
    if minority_count >= target_samples_per_class:
        print(" -> Kelas minoritas sudah memenuhi target. Beralih ke Undersampling Murni.")
        return _balance_with_undersampling(df)
    else:
        print(f" -> Kelas minoritas ({minority_count}) di bawah target ({target_samples_per_class}). Melakukan Oversampling.")
        if class_counts[0] > class_counts[1]: majority_label, minority_label = 0, 1
        else: majority_label, minority_label = 1, 0
        df_majority = df[df['label'] == majority_label]
        df_minority = df[df['label'] == minority_label]
        df_minority_oversampled = df_minority.sample(n=target_samples_per_class, replace=True, random_state=42)
        df_majority_undersampled = df_majority.sample(n=target_samples_per_class, random_state=42)
        return pd.concat([df_majority_undersampled, df_minority_oversampled]).sample(frac=1, random_state=42).reset_index(drop=True)

def process_reviews(df: pd.DataFrame, balancing_strategy: str = 'none', balance_ratio: float = 0.4) -> pd.DataFrame:
    """
    Memproses DataFrame mentah dengan strategi balancing yang bisa dipilih.
    balancing_strategy: 'none', 'undersample', 'oversample', 'hybrid'
    """
    print("Memulai pemrosesan data...")
    df_processed = df[['content', 'score']].copy()
    df_processed.rename(columns={'content': 'review'}, inplace=True)
    df_processed.dropna(subset=['review'], inplace=True)
    df_processed['label'] = df_processed['score'].apply(_map_sentiment)
    df_processed['cleaned_review'] = df_processed['review'].apply(_clean_text_for_bert)
    final_df = df_processed[['review', 'cleaned_review', 'score', 'label']].copy()
    
    if balancing_strategy == 'undersample':
        final_df = _balance_with_undersampling(final_df)
    elif balancing_strategy == 'oversample':
        final_df = _balance_with_oversampling(final_df)
    elif balancing_strategy == 'hybrid':
        final_df = _hybrid_balance_data(final_df, balance_ratio)
    else: # 'none'
        print("Tanpa penyeimbangan data. Menggunakan dataset asli (imbalanced).")

    print("\nDistribusi kelas final:")
    print(final_df['label'].value_counts())
    
    print(f"Total data akhir yang akan digunakan untuk training: {len(final_df)} ulasan.")
    
    print("Pemrosesan data selesai.")
    return final_df