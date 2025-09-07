"""
Project: Het vs. Homo Case Study — Texture Feature Analysis and Visualization
Author: Yasmin Kassim (analysis pipeline)
Description:
    End-to-end post-analysis pipeline that aggregates texture features from
    two genotype cohorts (Homo vs. Het), performs basic statistical
    comparisons, generates key visualizations (feature distributions, PCA,
    t-SNE), and trains a simple Random Forest classifier to assess
    separability. All artifacts (CSVs and figures) are written to a single
    output directory (`output_path`).

Inputs:
    - homo_path (str): Directory containing preprocessed crops for the Homo cohort.
    - het_path  (str): Directory containing preprocessed crops for the Het cohort.
    - output_path (str): Destination folder for ALL outputs (created if missing).

Upstream assumptions:
    - The helper module `homo_vs_het_Artur_texture_functions` provides:
        * process_genotype_folder(path, label) -> pd.DataFrame
          (returns one row per crop with columns including GLCM features and LBP bins)
        * compare_features(df, feature_list) -> None (prints/plots stats as implemented)
        * plot_feature_with_stats(df, feature_name) -> None (saves or shows a plot)
    - The combined feature table includes:
        * Column 'Genotype' with values like 'Homo' or 'Het'
        * GLCM columns: ['contrast', 'correlation', 'energy', 'homogeneity']
        * LBP histogram columns prefixed with 'LBP_bin_'

Outputs (all saved under `output_path`):
    - texture_features_glcm_lbp.csv
        Consolidated features for both cohorts (one row per crop).
    - PCA_Texture_Features.png
        2D PCA scatter of the texture feature space colored by genotype.
    - tSNE_Texture_Features.png
        2D t-SNE scatter for non-linear embedding visualization.
    - feature_importance.csv
        Random Forest feature importances for all included features.
    - feature_importance_top10.png
        Bar plot of the top-10 most informative features.

Pipeline overview:
    1) Load and featurize each genotype folder (via `process_genotype_folder`).
    2) Concatenate into a single DataFrame and persist as CSV.
    3) Print basic cohort counts and run univariate comparisons:
       - GLCM set: ['contrast', 'correlation', 'energy', 'homogeneity']
       - LBP bins: all columns starting with 'LBP_bin_'
    4) Plot per-feature distributions with stats overlays.
    5) Compute PCA (2D) and t-SNE (2D) for visualization; save plots.
    6) Train/evaluate a Random Forest classifier (stratified split, fixed seed),
       then export feature importance CSV and top-10 plot.

Dependencies:
    Python 3.9+ recommended
    pip install:
        numpy, pandas, scikit-learn, seaborn, matplotlib

Reproducibility & Notes:
    - Random states are fixed where applicable (e.g., TSNE, train/test split, RF).
    - All outputs are centralized in `output_path` for easier review and archiving.
    - If helper functions also save plots, they will write into the same `output_path`
      only if they are coded to do so; otherwise they may display inline.

Usage:
    - Set `homo_path`, `het_path`, and `output_path` near the top of the script.
    - Run: `python this_script.py`
    - Inspect artifacts in `output_path`.

Data:
    - Download the data from https://www.dropbox.com/scl/fo/yov2ksooppqkt5foqnbz9/AMoWgbrVeb35e-UqIOwAH6A?rlkey=zlvwwm9qwhjlx0m6dvgydgf1r&st=z0xf548c&dl=0

"""


import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from homo_vs_het_casestudy_functions import process_genotype_folder, compare_features, plot_feature_with_stats

#-------------- Paths where your processed files are
homo_path = r'C:\Users\Yasmin\Yasmin\Data\Eps8_project\Indzhykulian_Artur\Cdh23_v6j_Pcdh15_av3j\processed\homo_vs_het_casestudy\CDH23_KO_homo'
het_path  = r'C:\Users\Yasmin\Yasmin\Data\Eps8_project\Indzhykulian_Artur\Cdh23_v6j_Pcdh15_av3j\processed\homo_vs_het_casestudy\het_organized'

# Single output folder for EVERYTHING:
output_path = r'C:\Users\Yasmin\Yasmin\Data\Eps8_project\Indzhykulian_Artur\Cdh23_v6j_Pcdh15_av3j\processed\homo_vs_het_casestudy\results'
os.makedirs(output_path, exist_ok=True)

# Process Homo and Het folders
homo_features_df = process_genotype_folder(homo_path, 'Homo')
het_features_df  = process_genotype_folder(het_path, 'Het')

num_het  = het_features_df.shape[0]
num_homo = homo_features_df.shape[0]

print('########## Number of samples ################')
print(f"Number of Het cells (crops): {num_het}")
print(f"Number of Homo cells (crops): {num_homo}")

# Combine all features
all_features_df = pd.concat([homo_features_df, het_features_df], ignore_index=True)

# Save to CSV (in output_path)
csv_feat_path = os.path.join(output_path, 'texture_features_glcm_lbp.csv')
all_features_df.to_csv(csv_feat_path, index=False)
print(f"Saved all texture features to {csv_feat_path}")

##------------------------------------ Statistical Differences
# Define GLCM features
glcm_features = ['contrast', 'correlation', 'energy', 'homogeneity']
lbp_bins = [col for col in all_features_df.columns if col.startswith('LBP_bin_')]

print("\n--- GLCM Feature Comparisons ---")
compare_features(all_features_df, glcm_features)

print("\n--- LBP Histogram Comparisons ---")
compare_features(all_features_df, lbp_bins)

##------------------------------------------Visualization

# Plot all GLCM features with stats
for feature in glcm_features:
    plot_feature_with_stats(all_features_df, feature, output_path)

#------------------ PCA + t-SNE Visualization of Texture Features
feature_columns = glcm_features + [col for col in all_features_df.columns if col.startswith('LBP_bin_')]
X = all_features_df[feature_columns].values
y = all_features_df['Genotype'].values

# Use the single output folder
save_dir = output_path

# ----- PCA -----
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(8,6))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y, palette='Set2', s=60)
plt.title('PCA Projection of Texture Features')
plt.xlabel('PC1', fontsize=25)
plt.ylabel('PC2', fontsize=25)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.tight_layout()

pca_save_path = os.path.join(save_dir, 'PCA_Texture_Features.png')
plt.savefig(pca_save_path, dpi=300, bbox_inches='tight')
print(f"PCA plot saved to: {pca_save_path}")
plt.show()

# ----- t-SNE -----
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X)

plt.figure(figsize=(8,6))
sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], hue=y, palette='Set2', s=60)
plt.title('t-SNE Projection of Texture Features')
plt.xlabel('t-SNE 1', fontsize=25)
plt.ylabel('t-SNE 2', fontsize=25)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.legend(title='Genotype')
plt.tight_layout()

tsne_save_path = os.path.join(save_dir, 'tSNE_Texture_Features.png')
plt.savefig(tsne_save_path, dpi=300, bbox_inches='tight')
print(f"t-SNE plot saved to: {tsne_save_path}")
plt.show()

##--------------------- Random Forest classifier

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # 'Homo' -> 0, 'Het' -> 1

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print(f"Number of training samples: {len(X_train)}")
print(f"Number of testing samples: {len(X_test)}")

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Feature importances
importances = rf.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Save feature importance CSV (to output_path)
csv_save_path = os.path.join(save_dir, 'feature_importance.csv')
feature_importance_df.to_csv(csv_save_path, index=False)
print(f"Feature importance data saved to: {csv_save_path}")

# Plot top 10 features
plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10), palette='viridis')
plt.title('Top 10 Important Features in Random Forest')
plt.xlabel('Importance', fontsize=25)
plt.ylabel('Feature', fontsize=25)
plt.xticks(fontsize=20, rotation=90, ha='right')
plt.yticks(fontsize=20)
plt.tight_layout()

plot_save_path = os.path.join(save_dir, 'feature_importance_top10.png')
plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
print(f"Feature importance plot saved to: {plot_save_path}")
plt.show()
