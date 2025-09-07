"""
Project: WT vs KO Case Study — Bundle Height & Intensity Analysis
Author: Yasmin Kassim
Description:
    Pipeline to aggregate per-animal measurements from processed WT/KO folders,
    harmonize labels, compute normalized intensity metrics, generate summary
    statistics, and run pairwise tonotopic comparisons. All artifacts are saved
    to a single results directory (`output_path`).

Inputs:
    - folder_path: Root directory containing one subfolder per animal/region.
      Each subfolder is expected to include:
        • Intensity CSV:  .../Intensity_response/Allcells/Intensity_plots/option2.csv
            - must contain column: total_sum
        • Distance  CSV:  .../Distances/Physical_distances.csv
            - must contain columns: Distance, CLass
    - Genotype is inferred from the subfolder name:
        • 'Eps8KOControl' → KO
        • otherwise       → WT
    - Tonotopic region is inferred from the subfolder name: BASE / MIDDLE / APEX
    - Animal ID is extracted as 'MouseN' from the subfolder name.

Processing steps (high level):
    1) Load per-subfolder intensity and distance CSVs; enforce 1:1 row alignment.
    2) Build WT and KO DataFrames with columns:
       ['Total Intensity', 'Distance', 'CLass', 'Animal', 'tonotopic', 'KO_WT'].
    3) Normalize 'Total Intensity' globally to [0,1] → 'Normalized_total_intensities'.
    4) Clean class labels: OHC1/2/3 → OHC; derive 'Group' as CLass_WT/KO.
    5) Produce figures for distances and normalized intensities by tonotopic region/group.
    6) Compute summary tables and pairwise tonotopic tests; write all outputs to disk.

Outputs (all under `output_path`):
    - summary_table_figure9b_intensity_Height.csv
    - pairwise_Distance_ALL.csv
    - pairwise_Normalized_total_intensities_ALL.csv
    - Figure9a_pairwise_Welch_height.csv
    - Any plots/tables produced by:
        plot_distance_by_tonotopic_and_group()
        plot_Normalized_total_intensities_by_tonotopic_and_group()
        generate_summary_stats(save_groups=True)
        generate_normalized_intensity_summary()

Assumptions & safeguards:
    - CSVs exist and have the required columns; script raises with a clear error if not.
    - Row order/length in intensity and distance CSVs match within each subfolder.
    - Unknown CLass values are kept but excluded later when filtering.
    - All saves are directed to `output_path`; create it before running.

Dependencies:
    Python 3.9+; pandas; (project) WTvsKO_casestudy_functions module providing:
      collect_intensities, plot_distance_by_tonotopic_and_group,
      generate_summary_stats, plot_Normalized_total_intensities_by_tonotopic_and_group,
      pairwise_tonotopic_tests, generate_normalized_intensity_summary.

Usage:
    - Set `folder_path` to the processed data root and `output_path` to a results folder.
    - Run the script. Inspect printed summaries and CSV/figure outputs in `output_path`.

Data:
    - Download the data from here https://www.dropbox.com/scl/fo/e0ot1knh4nvuejqs30nsc/AIaELdkPp_BwN9Yi03x_wGE?rlkey=jjagsbitnvyu3kfj04ze6j60o&st=xumrvyy8&dl=0
"""



import re
import os
import pandas as pd
from WTvsKO_casestudy_functions import (
    collect_intensities,
    plot_distance_by_tonotopic_and_group,
    generate_summary_stats,
    plot_Normalized_total_intensities_by_tonotopic_and_group,
    pairwise_tonotopic_tests,
    generate_normalized_intensity_summary
)

# ----------------- Single output folder for EVERYTHING -----------------
output_path = r'C:\Users\Yasmin\Yasmin\work\My_papers\VASCilia\PLOS_review\Reviewer1\results'
os.makedirs(output_path, exist_ok=True)
os.chdir(output_path)  # <- makes any relative saves inside called functions land here
# ----------------------------------------------------------------------

folder_path = r'C:\Users\Yasmin\Yasmin\work\My_papers\VASCilia\PLOS_review\Reviewer1\processed_data_WT_KO'

df_KO = pd.DataFrame(columns=['Total Intensity', 'Distance', 'CLass', 'Animal', 'tonotopic','KO_WT'])
df_WT = pd.DataFrame(columns=['Total Intensity', 'Distance', 'CLass', 'Animal', 'tonotopic','KO_WT'])

for entry in os.listdir(folder_path):
    if entry == 'temp':
        continue
    pathtocsv_intensities = os.path.join(folder_path, entry, 'Intensity_response', 'Allcells', 'Intensity_plots', 'option2.csv')
    pathtocsv_distances = os.path.join(folder_path, entry, 'Distances', 'Physical_distances.csv')

    pattern1 = r"BASE"
    pattern2 = r"APEX"
    pattern3 = r"MIDDLE"

    match1 = re.search(pattern1, entry)
    match2 = re.search(pattern2, entry)
    match3 = re.search(pattern3, entry)

    if match1:
        tonotopic = match1.group(0)
    elif match2:
        tonotopic = match2.group(0)
    elif match3:
        tonotopic = match3.group(0)
    else:
        raise ValueError(f"No valid tonotopic region")

    pattern = r"Mouse(\d+)"
    match = re.search(pattern, entry)
    if match:
        animal = "Mouse" + match.group(1)
    else:
        animal = "Unknown"

    if 'Eps8KOControl' in entry:
        df_intensities, df_distances = collect_intensities(df_KO, pathtocsv_intensities, pathtocsv_distances)
        if 'total_sum' not in df_intensities.columns:
            raise KeyError(f"'total_sum' not found in {pathtocsv_intensities}. Columns: {list(df_intensities.columns)}")
        if 'Distance' not in df_distances.columns:
            raise KeyError(f"'Distance' not found in {pathtocsv_distances}. Columns: {list(df_distances.columns)}")
        if 'CLass' not in df_distances.columns:
            df_distances['CLass'] = 'Unknown'

        n = min(len(df_intensities), len(df_distances))
        if len(df_intensities) != len(df_distances):
            raise ValueError(
                f"[{entry}] Row count mismatch.\n"
                f"  intensities: {len(df_intensities)}  ({pathtocsv_intensities})\n"
                f"  distances:   {len(df_distances)}  ({pathtocsv_distances})\n"
                "Expected 1:1 rows (same bundles, same order)."
            )
        df_intensities = df_intensities.iloc[:n].reset_index(drop=True)
        df_distances  = df_distances.iloc[:n].reset_index(drop=True)

        newdata = pd.DataFrame({
            'Total Intensity': df_intensities['total_sum'],
            'Distance': df_distances['Distance'],
            'CLass': df_distances['CLass'],
            'Animal': animal,
            'tonotopic': tonotopic,
            'KO_WT': 'KO',
        })
        df_KO = pd.concat([df_KO, newdata], ignore_index=True)
    else:
        df_intensities, df_distances = collect_intensities(df_WT, pathtocsv_intensities, pathtocsv_distances)
        if 'total_sum' not in df_intensities.columns:
            raise KeyError(f"'total_sum' not found in {pathtocsv_intensities}. Columns: {list(df_intensities.columns)}")
        if 'Distance' not in df_distances.columns:
            raise KeyError(f"'Distance' not found in {pathtocsv_distances}. Columns: {list(df_distances.columns)}")
        if 'CLass' not in df_distances.columns:
            df_distances['CLass'] = 'Unknown'

        n = min(len(df_intensities), len(df_distances))
        if len(df_intensities) != len(df_distances):
            raise ValueError(
                f"[{entry}] Row count mismatch.\n"
                f"  intensities: {len(df_intensities)}  ({pathtocsv_intensities})\n"
                f"  distances:   {len(df_distances)}  ({pathtocsv_distances})\n"
                "Expected 1:1 rows (same bundles, same order)."
            )
        df_intensities = df_intensities.iloc[:n].reset_index(drop=True)
        df_distances  = df_distances.iloc[:n].reset_index(drop=True)

        newdata = pd.DataFrame({
            'Total Intensity': df_intensities['total_sum'],
            'Distance': df_distances['Distance'],
            'CLass': df_distances['CLass'],
            'Animal': animal,
            'tonotopic': tonotopic,
            'KO_WT': 'WT',
        })
        df_WT = pd.concat([df_WT, newdata], ignore_index=True)

# Replace 'OHC1', 'OHC2', 'OHC3', etc., with 'OHC' in the 'Class' column
df_WT['CLass'] = df_WT['CLass'].str.replace(r'OHC\d+', 'OHC', regex=True)
df_KO['CLass'] = df_KO['CLass'].str.replace(r'OHC\d+', 'OHC', regex=True)

df_WT['Group'] = df_WT['CLass'].apply(lambda x: x + '_WT')
df_KO['Group'] = df_KO['CLass'].apply(lambda x: x + '_KO')

df_combined = pd.concat([df_WT, df_KO], ignore_index=True)

df_combined_filtered = df_combined[df_combined['CLass'] != 'Unknown'].copy()

global_min = df_combined_filtered['Total Intensity'].min()
global_max = df_combined_filtered['Total Intensity'].max()
den = global_max - global_min
if den == 0:
    print("[WARN] All Total Intensity values identical; setting normalized intensities to 0.")
    df_combined_filtered['Normalized_total_intensities'] = 0.0
else:
    df_combined_filtered['Normalized_total_intensities'] = (
        df_combined_filtered['Total Intensity'] - global_min
    ) / den

#---------------------------------------------------
plot_distance_by_tonotopic_and_group(df_combined_filtered, output_path)

summary = generate_summary_stats(
    df_combined_filtered,
    save_groups=True,
    out_dir=output_path,   # save per-group CSVs here
    cols_to_save=['ID', 'Distance', 'KO_WT', 'tonotopic', 'CLass']
)
print(summary)

plot_Normalized_total_intensities_by_tonotopic_and_group(df_combined_filtered, output_path)

summary_df = generate_normalized_intensity_summary(df_combined_filtered, out_dir=output_path)
print(summary_df)
summary_df.to_csv(os.path.join(output_path, 'summary_table_figure9b_intensity_Height.csv'), index=False)

# Example: show WT × IHC only
OUTDIR = output_path
INCLUDE_BASE_VS_APEX = True

os.makedirs(OUTDIR, exist_ok=True)

# --- 1) HEIGHTS (Distance) ---
tests_height = pairwise_tonotopic_tests(
    df_combined_filtered,
    value_col="Distance",
    include_base_vs_apex=INCLUDE_BASE_VS_APEX
)
tests_height.to_csv(os.path.join(OUTDIR, "pairwise_Distance_ALL.csv"), index=False)

for geno in ["WT", "KO"]:
    for cell in ["IHC", "OHC"]:
        sub = tests_height.query("KO_WT == @geno and CLass == @cell").copy()
        if sub.empty:
            continue
        print(f"\n=== Distance :: {geno}/{cell} ===")
        print(sub[['level1','level2','n1','n2','mean1','mean2','mean_diff',
                   'ci_low','ci_high','t','dof','p','p_holm','g']].to_string(index=False))
        #sub.to_csv(os.path.join(OUTDIR, f"pairwise_Distance_{geno}_{cell}.csv"), index=False)

# --- 2) INTENSITIES (Normalized_total_intensities) ---
tests_int = pairwise_tonotopic_tests(
    df_combined_filtered,
    value_col="Normalized_total_intensities",
    include_base_vs_apex=INCLUDE_BASE_VS_APEX
)
tests_int.to_csv(os.path.join(OUTDIR, "pairwise_Normalized_total_intensities_ALL.csv"), index=False)

for geno in ["WT", "KO"]:
    for cell in ["IHC", "OHC"]:
        sub = tests_int.query("KO_WT == @geno and CLass == @cell").copy()
        if sub.empty:
            continue
        print(f"\n=== Normalized_total_intensities :: {geno}/{cell} ===")
        print(sub[['level1','level2','n1','n2','mean1','mean2','mean_diff',
                   'ci_low','ci_high','t','dof','p','p_holm','g']].to_string(index=False))
        #sub.to_csv(os.path.join(OUTDIR, f"pairwise_Normalized_total_intensities_{geno}_{cell}.csv"), index=False)

# Also save the convenience CSV here
tests_height.to_csv(os.path.join(output_path, "Figure9a_pairwise_Welch_height.csv"), index=False)

print('done')
