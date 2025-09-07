import os
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte
import numpy as np
from scipy.stats import mannwhitneyu
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def compare_features(df, feature_columns):
    for feature in feature_columns:
        homo_values = df[df['Genotype'] == 'Homo'][feature]
        het_values = df[df['Genotype'] == 'Het'][feature]

        u_stat, p_val = mannwhitneyu(homo_values, het_values, alternative='two-sided')
        print(f'{feature}: U={u_stat}, p-value={p_val:.4f}')
def extract_glcm_features(image, distances=[1, 2, 3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    glcm = graycomatrix(image, distances=distances, angles=angles, symmetric=True, normed=True)
    features = {}
    for prop in ['contrast', 'correlation', 'energy', 'homogeneity']:
        features[prop] = graycoprops(glcm, prop).mean()
    return features

def extract_lbp_features(image, P=8, R=1):
    lbp = local_binary_pattern(image, P, R, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)  # Normalize
    return hist

def process_genotype_folder(root_path, genotype_label):
    all_features = []

    for sample in os.listdir(root_path):
        sample_path = os.path.join(root_path, sample, 'measurements', 'crops')
        if not os.path.exists(sample_path):
            print(f"Skipping {sample}: No measurements/crops folder found.")
            continue

        for file in os.listdir(sample_path):
            if file.startswith('raw_crop') and file.endswith('.npy'):
                crop_path = os.path.join(sample_path, file)
                raw_crop = np.load(crop_path)

                # Convert to grayscale (average over last channel if RGB)
                if raw_crop.ndim == 4:
                    raw_crop = raw_crop[..., :3]  # In case there are more channels
                    raw_crop_gray = rgb2gray(raw_crop.mean(axis=0))  # Average along Z
                elif raw_crop.ndim == 3:
                    raw_crop_gray = rgb2gray(raw_crop.mean(axis=0))
                else:
                    print(f"Unexpected shape in {crop_path}")
                    continue
                raw_crop_gray = raw_crop_gray - raw_crop_gray.min()
                if raw_crop_gray.max() != 0:
                    raw_crop_gray = raw_crop_gray / raw_crop_gray.max()

                raw_crop_gray = img_as_ubyte(raw_crop_gray)

                # Extract features
                # plt.imshow(raw_crop_gray)
                # plt.show()
                glcm_feats = extract_glcm_features(raw_crop_gray)
                lbp_hist = extract_lbp_features(raw_crop_gray)

                features = {
                    'Genotype': genotype_label,
                    'Sample': sample,
                    'CropFile': file,
                    **glcm_feats
                }

                # Add LBP histogram bins
                for i, val in enumerate(lbp_hist):
                    features[f'LBP_bin_{i}'] = val

                all_features.append(features)

    return pd.DataFrame(all_features)


##------------------------------------------Visualization
def plot_feature_with_stats(df, feature, save_dir):
    plt.figure(figsize=(8, 5))
    ax = sns.violinplot(data=df, x='Genotype', y=feature, palette='Set2', inner='box')

    stats = df.groupby('Genotype')[feature].agg(['mean', 'median']).reset_index()
    print(df.groupby('Genotype')[feature].median())
    for idx, genotype in enumerate(df['Genotype'].unique()):
        median_val = stats.loc[stats['Genotype'] == genotype, 'median'].values[0]

        ax.annotate(f"Median: {median_val:.3f}",
                    xy=(idx, median_val),
                    xytext=(0, 5),  # 5 points above
                    textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=14, color='blue', weight='bold')

    #plt.title(f'{feature} by Genotype')
    plt.xlabel('Genotype', fontsize=25)
    plt.ylabel(feature[:1].upper() + feature[1:], fontsize=25)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f'{feature}_by_Genotype.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure: {save_path}")
    plt.show()



def plot_feature(df, feature):
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x='Genotype', y=feature, palette='Set2')
    #plt.title(f'{feature} by Genotype', fontsize=18)
    plt.xlabel('Genotype', fontsize=25)
    plt.ylabel(feature[:1].upper() + feature[1:], fontsize=25)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.tight_layout()
