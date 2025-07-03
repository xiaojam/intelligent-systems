import pandas as pd
from google_play_scraper import reviews, Sort

def scrape_google_play_reviews(app_id, lang='id', country='id', count=50000):
    """
    Mengambil ulasan dari Google Play Store.
    """
    print(f"Memulai proses crawling untuk {count} ulasan dari aplikasi: {app_id}.")
    print("Proses ini mungkin memakan waktu sangat lama, mohon bersabar...")
    
    result, _ = reviews(
        app_id,
        lang=lang,
        country=country,
        sort=Sort.NEWEST,
        count=count,
        filter_score_with=None
    )

    if not result:
        print("Tidak ada ulasan yang ditemukan atau terjadi error.")
        return pd.DataFrame()

    df = pd.DataFrame(result)
    df = df[['at', 'score', 'content']]
    df.rename(columns={'at': 'date', 'score': 'rating', 'content': 'text'}, inplace=True)
    
    print(f"Crawling selesai. Berhasil mengambil {len(df)} ulasan.")
    return df