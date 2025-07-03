import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from data_crawler import crawl_reviews
from text_processor import process_reviews
from sentiment_model import SentimentAnalyzer

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
        print(f"Data untuk {app_name} siap digunakan.")
    return processed_dataframes

def run_training_pipeline(processed_dataframes: dict, epochs: int = 3) -> dict:
    evaluation_results = {}
    for app_name, df in processed_dataframes.items():
        print(f"\n{'='*50}\nMemulai Training Model untuk: {app_name}\n{'='*50}\n")
        if len(df) < 100:
            print(f"Data tidak cukup untuk {app_name}, lanjut.")
            evaluation_results[app_name] = {'accuracy': 0, 'f1_macro': 0, 'report_dict': {}, 'confusion_matrix': np.zeros((2,2))}
            continue
        analyzer = SentimentAnalyzer()
        analyzer.prepare_dataloaders(df)
        analyzer.train(epochs=epochs)
        model_save_path = f"models/model_{app_name.lower().replace(' ', '_')}.pt"
        analyzer.save_model(model_save_path)
        results = analyzer.evaluate()
        evaluation_results[app_name] = results
    return evaluation_results

def plot_confusion_matrix(cm, class_names, app_name):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=class_names, yticklabels=class_names)
    ax.set_title(f'Confusion Matrix for {app_name}', fontsize=16, pad=20)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{app_name.lower().replace(" ", "_")}.png', dpi=300)
    plt.show()

def generate_comparative_visuals(processed_dataframes: dict, evaluation_results: dict):
    print(f"\n{'='*50}\nMembuat Visualisasi Komparatif\n{'='*50}\n")
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Visualisasi Distribusi Sentimen
    sentiment_counts = {app_name: df['label'].value_counts(normalize=True) * 100 for app_name, df in processed_dataframes.items()}
    df_sentiment = pd.DataFrame(sentiment_counts).T.fillna(0).rename(columns={1: 'Positive (%)', 0: 'Negative (%)'})
    ax1 = df_sentiment.plot(kind='bar', figsize=(12, 7), rot=0, color={'Positive (%)': 'forestgreen', 'Negative (%)': 'crimson'})
    ax1.set_title('Sentiment Distribution of Dating Apps Reviews', fontsize=16, pad=20)
    ax1.set_xlabel('Dating App', fontsize=12)
    ax1.set_ylabel('Percentage (%)', fontsize=12)
    ax1.legend(title='Sentiment')
    for container in ax1.containers: ax1.bar_label(container, fmt='%.1f%%')
    plt.tight_layout()
    plt.savefig('sentiment_distribution_comparison.png', dpi=300)
    plt.show()

    # Visualisasi F1-Score (Macro)
    f1_scores = {app_name: result.get('f1_macro', 0) for app_name, result in evaluation_results.items()}
    df_f1 = pd.DataFrame(list(f1_scores.items()), columns=['Application', 'F1-Score (Macro)'])
    plt.figure(figsize=(10, 6))
    ax2 = sns.barplot(x='Application', y='F1-Score (Macro)', data=df_f1, palette='plasma')
    ax2.set_title('Model Performance (F1-Score Macro) per App Dataset', fontsize=16, pad=20)
    ax2.set_xlabel('Dating App', fontsize=12)
    ax2.set_ylabel('F1-Score (Macro)', fontsize=12)
    ax2.set_ylim(0, 1)
    for container in ax2.containers: ax2.bar_label(container, fmt='%.3f')
    plt.tight_layout()
    plt.savefig('model_performance_comparison.png', dpi=300)
    plt.show()

    # Plot Confusion Matrix untuk setiap aplikasi
    for app_name, result in evaluation_results.items():
        if 'confusion_matrix' in result and result['confusion_matrix'].any():
            plot_confusion_matrix(result['confusion_matrix'], ['Negative', 'Positive'], app_name)