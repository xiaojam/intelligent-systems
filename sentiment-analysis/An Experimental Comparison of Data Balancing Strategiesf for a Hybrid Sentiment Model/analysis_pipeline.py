# analysis_pipeline.py

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
            print(f"File '{app_id}_reviews.csv' ditemukan, memuat data lokal.")
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
        history = analyzer.train(epochs=epochs) 
        
        # Plot training history untuk setiap model
        plot_training_history(history, app_name, model_type)
        
        model_save_path = f"models/{app_name.lower()}_{model_type}.pt"
        analyzer.save_model(model_save_path)
        results = analyzer.evaluate()
        evaluation_results[app_name] = results
    return evaluation_results

def generate_comparative_visuals(all_results: dict, processed_dataframes: dict):
    """
    Membuat visualisasi komparatif, termasuk perbandingan metrik F1-Score non-macro.
    """
    print(f"\n{'='*60}\nMembuat Visualisasi Komparatif\n{'='*60}\n")
    plt.style.use('seaborn-v0_8-whitegrid')

    # 1. Menyiapkan DataFrame Hasil yang lebih detail
    results_list = []
    for model_type, app_results in all_results.items():
        for app_name, metrics in app_results.items():
            if metrics and 'report_dict' in metrics:
                report = metrics['report_dict']
                results_list.append({
                    'Aplikasi': app_name, 'Model': model_type.upper(),
                    'Tipe F1-Score': 'Macro', 'Skor': metrics.get('f1_macro', 0)
                })
                results_list.append({
                    'Aplikasi': app_name, 'Model': model_type.upper(),
                    'Tipe F1-Score': 'Positif', 'Skor': report['Positif']['f1-score']
                })
                results_list.append({
                    'Aplikasi': app_name, 'Model': model_type.upper(),
                    'Tipe F1-Score': 'Negatif', 'Skor': report['Negatif']['f1-score']
                })
    df_results = pd.DataFrame(results_list)

    # 2. Visualisasi Perbandingan Kinerja F1-Score (Macro, Positif, Negatif)
    g = sns.catplot(
        data=df_results, kind="bar",
        x="Aplikasi", y="Skor", hue="Tipe F1-Score", col="Model",
        palette="muted", height=6, aspect=0.8
    )
    g.despine(left=True)
    g.set_axis_labels("Aplikasi", "F1-Score")
    g.set_titles("Kinerja Model {col_name}")
    g.set(ylim=(0.7, 1.0))
    g.tight_layout()
    g.savefig('model_performance_detailed_comparison.png', dpi=300)
    plt.show()

    # 3. Visualisasi Distribusi Sentimen
    sentiment_counts = {app_name: df['label'].value_counts(normalize=True) * 100 for app_name, df in processed_dataframes.items()}
    df_sentiment = pd.DataFrame(sentiment_counts).T.fillna(0).rename(columns={1: 'Positif (%)', 0: 'Negatif (%)'})
    ax1 = df_sentiment.plot(kind='bar', figsize=(10, 6), rot=0, color={'Positif (%)': 'forestgreen', 'Negatif (%)': 'crimson'})
    ax1.set_title('Distribusi Sentimen Ulasan Pengguna', fontsize=16)
    ax1.set_xlabel('Aplikasi'); ax1.set_ylabel('Persentase (%)')
    for container in ax1.containers: ax1.bar_label(container, fmt='%.1f%%')
    plt.tight_layout(); plt.savefig('sentiment_distribution.png', dpi=300); plt.show()
    
    # 4. Visualisasi Confusion Matrix untuk setiap hasil
    print(f"\n{'='*60}\nVisualisasi Confusion Matrix per Eksperimen\n{'='*60}\n")
    for model_type, app_results in all_results.items():
        for app_name, metrics in app_results.items():
            if metrics and 'confusion_matrix' in metrics:
                title = f'Confusion Matrix: {app_name} ({model_type.upper()})'
                plot_confusion_matrix(metrics['confusion_matrix'], ['Negatif', 'Positif'], title)