import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_crawler import crawl_reviews
from text_processor import process_reviews
from sentiment_model import SentimentAnalyzer

def plot_confusion_matrix(cm, class_names, title):
    """Helper function untuk membuat plot heatmap confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '_')).rstrip().replace(" ", "_")
    plt.savefig(f'cm_{safe_title}.png', dpi=300, bbox_inches='tight')
    plt.show()

def run_data_pipeline(apps_to_analyze: list, target_review_count: int) -> dict:
    processed_dataframes = {}
    for app in apps_to_analyze:
        app_id, app_name = app['id'], app['name']
        print(f"\n{'='*50}\nMemulai Proses Data untuk: {app_name}\n{'='*50}\n")
        try:
            df_raw = pd.read_csv(f"{app_id}_reviews.csv")
            print(f"📄 File '{app_id}_reviews.csv' ditemukan, memuat data lokal.")
        except FileNotFoundError:
            df_raw = crawl_reviews(app_id=app_id, target_count=target_review_count)
        df_processed = process_reviews(df_raw)
        processed_dataframes[app_name] = df_processed
    return processed_dataframes

def run_training_pipeline(processed_dataframes: dict, model_type: str, epochs: int) -> dict:
    evaluation_results = {}
    for app_name, df in processed_dataframes.items():
        print(f"\n{'='*50}\nTraining Model {model_type.upper()} untuk: {app_name}\n{'='*50}\n")
        if len(df) < 100:
            print(f"Data tidak cukup untuk {app_name}, lanjut."); evaluation_results[app_name] = {}; continue
        
        analyzer = SentimentAnalyzer(model_type=model_type)
        analyzer.prepare_dataloaders(df)
        analyzer.train(epochs=epochs)
        model_save_path = f"models/{app_name.lower()}_{model_type}.pt"
        analyzer.save_model(model_save_path)
        results = analyzer.evaluate()
        evaluation_results[app_name] = results
    return evaluation_results

def generate_comparative_visuals(all_results: dict, processed_dataframes: dict):
    print(f"\n{'='*50}\nMembuat Visualisasi Komparatif\n{'='*50}\n")
    plt.style.use('seaborn-v0_8-whitegrid')

    # 1. Visualisasi Perbandingan Kinerja F1-Score
    results_list = []
    for model_type, app_results in all_results.items():
        for app_name, metrics in app_results.items():
            if metrics:
                res = {'Aplikasi': app_name, 'Model': model_type.upper(), 'F1-Score (Macro)': metrics.get('f1_macro', 0)}
                results_list.append(res)
    df_results = pd.DataFrame(results_list)

    plt.figure(figsize=(12, 8))
    ax = sns.barplot(data=df_results, x='Aplikasi', y='F1-Score (Macro)', hue='Model', palette='viridis')
    ax.set_title('Perbandingan Kinerja F1-Score (Macro) Antar Model', fontsize=16)
    ax.set_xlabel('Aplikasi', fontsize=12); ax.set_ylabel('F1-Score (Macro)', fontsize=12); ax.set_ylim(0.7, 1.0)
    for container in ax.containers: ax.bar_label(container, fmt='%.3f')
    plt.legend(title='Tipe Model'); plt.tight_layout(); plt.savefig('model_performance_comparison.png', dpi=300); plt.show()

    # 2. Visualisasi Distribusi Sentimen
    sentiment_counts = {app_name: df['label'].value_counts(normalize=True) * 100 for app_name, df in processed_dataframes.items()}
    df_sentiment = pd.DataFrame(sentiment_counts).T.fillna(0).rename(columns={1: 'Positif (%)', 0: 'Negatif (%)'})
    ax1 = df_sentiment.plot(kind='bar', figsize=(10, 6), rot=0, color={'Positif (%)': 'forestgreen', 'Negatif (%)': 'crimson'})
    ax1.set_title('Distribusi Sentimen Ulasan Pengguna', fontsize=16)
    ax1.set_xlabel('Aplikasi', fontsize=12); ax1.set_ylabel('Persentase (%)', fontsize=12)
    for container in ax1.containers: ax1.bar_label(container, fmt='%.1f%%')
    plt.tight_layout(); plt.savefig('sentiment_distribution.png', dpi=300); plt.show()
    
    # 3. Visualisasi Confusion Matrix untuk setiap hasil
    print(f"\n{'='*50}\nVisualisasi Confusion Matrix per Eksperimen\n{'='*50}\n")
    for model_type, app_results in all_results.items():
        for app_name, metrics in app_results.items():
            if metrics and 'confusion_matrix' in metrics:
                title = f'Confusion Matrix: {app_name} ({model_type.upper()})'
                plot_confusion_matrix(metrics['confusion_matrix'], ['Negatif', 'Positif'], title)