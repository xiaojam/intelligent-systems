import matplotlib.pyplot as plt
import numpy as np

def plot_sentiment_distribution(distribution, title=""):
    """
    Membuat pie chart untuk distribusi sentimen 5 level.
    """
    labels = distribution.index
    sizes = distribution.values
    
    order = ['Sangat Positif', 'Positif', 'Netral', 'Negatif', 'Sangat Negatif']
    color_map = {
        'Sangat Positif': '#2E7D32', 
        'Positif': '#66BB6A', 
        'Netral': '#FFC107',
        'Negatif': '#EF5350', 
        'Sangat Negatif': '#C62828'
    }
    
    # Melakukan filter dan pengurutan data yang ada
    filtered_labels = [label for label in order if label in labels]
    filtered_sizes = [distribution[label] for label in filtered_labels]
    filtered_colors = [color_map[label] for label in filtered_labels]

    plt.style.use('seaborn-v0_8-deep')
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.pie(filtered_sizes, labels=filtered_labels, autopct='%1.1f%%',
           startangle=90, colors=filtered_colors, textprops={'fontsize': 12})
    ax.axis('equal')  
    plt.title(title, size=16)
    plt.savefig(f"{title.replace(' ', '_').lower()}.png")
    plt.show()

def plot_radar_chart(scores, title="Balanced Scorecard Kinerja"):
    """
    Membuat radar chart dari skor aspek.
    """
    labels, stats = list(scores.keys()), list(scores.values())
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    stats += stats[:1]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))
    ax.plot(angles, stats, color='darkviolet', linewidth=2)
    ax.fill(angles, stats, color='darkviolet', alpha=0.25)
    ax.set_yticklabels([])
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=11)
    plt.title(title, size=20, color='darkviolet', y=1.1)
    for angle, stat in zip(angles[:-1], stats[:-1]):
        ax.text(angle, stat + 5, f"{stat:.1f}", ha='center', va='center', fontsize=12)
    plt.savefig(f"{title.replace(' ', '_').lower()}.png")
    plt.show()

def plot_dynamic_trend(df_dynamic, title="Tren Kinerja BSC"):
    """
    Membuat line chart untuk tren skor dari waktu ke waktu.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 8))
    for column in df_dynamic.columns:
        ax.plot(df_dynamic.index.astype(str), df_dynamic[column], marker='o', linestyle='-', label=column)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Periode (Bulan-Tahun)", fontsize=12)
    ax.set_ylabel("Skor Kinerja (0-100)", fontsize=12)
    ax.legend(title="Aspek Kinerja", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_').lower()}.png")
    plt.show()