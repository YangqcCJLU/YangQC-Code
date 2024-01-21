import matplotlib.pyplot as plt
import numpy as np

# Classification accuracy
data = {
    'SVM': {'None': [93.6, 90.3, 98.9, 60.6, 95.7, 87.6],
            'PCA': [92.1, 78.4, 98.0, 43.9, 92.1, 80.5],
            'iRF': [97.0, 89.5, 85.2, 97.6, 94.0, 92.3],
            'SPA': [94.0, 89.7, 98.8, 68.2, 95.6, 89.1]},

    'ELM': {'None': [97.6, 89.7, 95.3, 78.6, 95.6, 91.0],
            'PCA': [94.6, 75.3, 91.3, 61.9, 90.8, 82.2],
            'iRF': [98.7, 89.4, 85.9, 97.3, 93.6, 92.6],
            'SPA': [98.2, 91.3, 95.5, 75.7, 95.5, 90.9]},

    '1DCNN': {'None': [99.4, 91.6, 97.1, 96.2, 93.7, 95.5],
              'PCA': [88.9, 58.6, 99.3, 78.0, 41.4, 73.5],
              'iRF': [98.8, 92.7, 91.0, 96.8, 89.5, 93.6],
              'SPA': [94.4, 61.8, 98.3, 91.5, 93.1, 87.6]},

    '1DCNN+CAM': {'None': [99.5, 90.5, 98.1, 98.6, 95.1, 96.3],
                  'PCA': [91.3, 70.7, 96.1, 76.3, 61.5, 79.3],
                  'iRF': [98.9, 91.0, 92.1, 97.7, 91.0, 94.0],
                  'SPA': [93.3, 91.8, 97.2, 74.2, 92.7, 89.7]}
}

# List of categories and models
categories = ['none', 'procymidone', 'oxytetracycline', 'indoleacetic acid', 'gibberellin', 'overall']
models = ['SVM', 'ELM', '1DCNN', '1DCNN+CAM']

# Plot histograms
bar_width = 0.16
gap = 0.04  # Adjust spacing between bars
index = np.arange(len(categories))

plt.figure(figsize=(10, 6))
for model in models:
    plt.bar(index, data[model]['None'], bar_width, label=model, alpha=0.8)
    index = index + bar_width + gap

plt.xlabel('Categories', fontsize=14)
plt.ylabel('Accuracy (%)', fontsize=14)
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.xticks(index - bar_width * (len(models) / 1.2), categories)
plt.legend(loc='upper right', fontsize=12, bbox_to_anchor=(1.25, 1))
plt.tight_layout()
plt.savefig("bar.tif", dpi=300, bbox_inches='tight')
plt.show()
