import pandas as pd
from google_play_scraper import reviews, Sort
import time

def crawl_reviews(app_id: str, target_count: int = 500, lang: str = 'id', country: str = 'id') -> pd.DataFrame:
    """
    Melakukan crawling review dari Google Play Store untuk aplikasi tertentu.

    Args:
        app_id (str): ID aplikasi (contoh: 'com.tokopedia').
        target_count (int): Jumlah maksimum review yang ingin diambil.
        lang (str): Bahasa review.
        country (str): Negara/region review.

    Returns:
        pd.DataFrame: DataFrame berisi review yang berhasil di-crawl.
    """
    print(f"Start crawling data aplikasi: {app_id}")
    
    all_reviews = []
    continuation_token = None
    
    while len(all_reviews) < target_count:
        # Maksimum count per request adalah 200
        result, token = reviews(
            app_id,
            lang=lang,
            country=country,
            sort=Sort.NEWEST,
            count=200,
            continuation_token=continuation_token
        )
        
        if not result:
            print("Tidak ada ulasan lebih lanjut yang ditemukan.")
            break
            
        all_reviews.extend(result)
        continuation_token = token
        
        print(f"Terkumpul: {len(all_reviews)} ulasan")
        time.sleep(1) 

    print(f"Selesai! Berhasil mendapatkan {len(all_reviews)} ulasan untuk {app_id}.")
    
    df_reviews = pd.DataFrame(all_reviews)
    
    # Simpan ke CSV sebagai backup
    nama_file_csv = f'{app_id}_reviews.csv'
    df_reviews.to_csv(nama_file_csv, index=False)
    print(f"Data crawling disimpan di '{nama_file_csv}'")
    
    return df_reviews