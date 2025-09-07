import seaborn as sns
import matplotlib.pyplot as plt
from typing import Optional, List, Union
from pathlib import Path
import re
import os
import numpy as np
import pandas as pd
from statsmodels.stats.weightstats import ttest_ind, DescrStatsW, CompareMeans


# # ---------- 1) Helper: Holm adjustment (no extra libs needed) ----------
def holm_adjust(pvals):
    """Holm-Bonferroni adjusted p-values (two-sided)."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    p_sorted = p[order]
    adj_sorted = (m - np.arange(m)) * p_sorted
    # enforce monotonicity and cap at 1
    adj_sorted = np.maximum.accumulate(adj_sorted)
    adj_sorted = np.minimum(adj_sorted, 1.0)
    adj = np.empty_like(adj_sorted)
    adj[order] = adj_sorted
    return adj

# # ---------- 2) Build the label used in your screenshot ----------
def build_label(s):
    # e.g., "WT_Base_IHC"
    return f"{s['KO_WT'].upper()}_{s['tonotopic'].capitalize()}_{s['CLass'].upper()}"


def generate_summary_stats(
    df: pd.DataFrame,
    save_groups: bool = True,
    out_dir: Union[str, os.PathLike] = "group_csvs",
    cols_to_save: Optional[List[str]] = None,   # e.g. ['ID','Distance','KO_WT','tonotopic','CLass']
) -> pd.DataFrame:
    """
    Builds a combined group key KO_WT + tonotopic(capitalized) + CLass,
    returns a summary table, and (optionally) writes one CSV per group with all rows.
    Robust to missing columns in `cols_to_save`. Compatible with Python 3.7+.
    """

    # Work on a copy to avoid side-effects
    df = df.copy()

    # Ensure strings are clean and build the group key
    df['Tonotopic_KO_WT_CLass'] = (
        df['KO_WT'].astype(str).str.strip() + '_' +
        df['tonotopic'].astype(str).str.strip().str.capitalize() + '_' +
        df['CLass'].astype(str).str.strip()
    )

    # Group and compute stats (Pandas std = sample SD, ddof=1)
    summary_df = df.groupby('Tonotopic_KO_WT_CLass', sort=True).agg(
        mean_height=('Distance', 'mean'),
        std_height=('Distance', 'std'),
        median_height=('Distance', 'median'),
        count=('Distance', 'count')
    ).reset_index()

    # Round for cleaner table
    summary_df = summary_df.round({
        'mean_height': 2,
        'std_height': 2,
        'median_height': 2
    })

    # Optionally save one CSV per group with all rows
    if save_groups:
        out_dir_path = Path(out_dir)
        out_dir_path.mkdir(parents=True, exist_ok=True)

        # Decide which columns to write
        if cols_to_save is None:
            cols_requested = list(df.columns)  # everything (incl. group key)
        else:
            # keep order, dedupe, and ensure the group key is present
            cols_requested = []
            for c in cols_to_save + ['Tonotopic_KO_WT_CLass']:
                if c not in cols_requested:
                    cols_requested.append(c)

        # Keep only columns that actually exist; warn about missing
        existing_cols = set(df.columns)
        cols_to_write = [c for c in cols_requested if c in existing_cols]
        missing = [c for c in cols_requested if c not in existing_cols]
        if missing:
            print(f"[generate_summary_stats] Warning: missing columns skipped: {missing}")

        subdir = out_dir_path / 'Figure9a_length_csvfiles'
        subdir.mkdir(parents=True, exist_ok=True)

        # ...
        for gname, gdf in df.groupby('Tonotopic_KO_WT_CLass', sort=True):
            safe = re.sub(r'[^A-Za-z0-9._-]+', '_', gname)
            gdf.loc[:, cols_to_write].to_csv(subdir / f"{safe}.csv", index=False)

        df.loc[:, cols_to_write].to_csv(subdir / "ALL_GROUPS.csv", index=False)

    return summary_df




def generate_normalized_intensity_summary(
    df: pd.DataFrame,
    save_groups: bool = True,
    out_dir: Union[str, os.PathLike] = "group_csvs",
    cols_to_save: Optional[List[str]] = None,  # e.g. ['ID','frame','KO_WT','tonotopic','CLass','Normalized_total_intensities']
) -> pd.DataFrame:
    """
    Builds a combined group key KO_WT + tonotopic(capitalized) + CLass,
    summarizes Normalized_total_intensities, and (optionally) writes one CSV per group.
    Excludes 'Total Intensity' and 'Distance' columns from the saved per-group CSVs.
    """

    # Work on a copy to avoid side-effects
    df = df.copy()

    # Group key
    df['Tonotopic_KO_WT_CLass'] = (
        df['KO_WT'].astype(str).str.strip() + '_' +
        df['tonotopic'].astype(str).str.strip().str.capitalize() + '_' +
        df['CLass'].astype(str).str.strip()
    )

    # numeric & drop NaNs for the value column
    df['Normalized_total_intensities'] = pd.to_numeric(df['Normalized_total_intensities'], errors='coerce')
    df = df.dropna(subset=['Normalized_total_intensities'])

    # Summary stats
    summary_df = df.groupby('Tonotopic_KO_WT_CLass', sort=True).agg(
        mean_value=('Normalized_total_intensities', 'mean'),
        std_value=('Normalized_total_intensities', 'std'),
        median_value=('Normalized_total_intensities', 'median'),
        count=('Normalized_total_intensities', 'count')
    ).reset_index().round({'mean_value': 2, 'std_value': 2, 'median_value': 2})

    if save_groups:
        out_dir_path = Path(out_dir)
        out_dir_path.mkdir(parents=True, exist_ok=True)

        # Always exclude these from saved per-group CSVs
        excluded_cols = {"Total Intensity", "Distance"}

        # Decide which columns to request
        if cols_to_save is None:
            cols_requested = list(df.columns)  # everything (incl. group key)
        else:
            # keep order, dedupe, ensure group key present
            cols_requested = []
            for c in cols_to_save + ['Tonotopic_KO_WT_CLass']:
                if c not in cols_requested:
                    cols_requested.append(c)

        # Keep only existing columns and drop excluded
        existing_cols = set(df.columns)
        cols_to_write = [c for c in cols_requested if c in existing_cols and c not in excluded_cols]

        # Warn if user requested missing columns
        missing = [c for c in cols_requested if c not in existing_cols]
        if missing:
            print(f"[generate_normalized_intensity_summary] Warning: missing columns skipped: {missing}")

        subdir = out_dir_path / 'Figure9b_intensity_csvfiles'
        subdir.mkdir(parents=True, exist_ok=True)

        # Save one CSV per group
        for gname, gdf in df.groupby('Tonotopic_KO_WT_CLass', sort=True):
            safe = re.sub(r'[^A-Za-z0-9._-]+', '_', gname)
            gdf.loc[:, cols_to_write].to_csv(subdir / f"{safe}.csv", index=False)

        df.loc[:, cols_to_write].to_csv(subdir / "ALL_GROUPS.csv", index=False)

    return summary_df


def plot_distance_by_tonotopic_and_group(df, output_path):
    # Create a new column combining 'Tonotopic' and 'Group' for clear labeling on the plot
    df['Tonotopic_KO_WT_CLass'] = df['KO_WT'] + '_' + df['tonotopic'].str.capitalize() + '_' + df['CLass']

    # Explicitly define the order of the groups
    full_order = ['WT_Base_IHC', 'WT_Middle_IHC', 'WT_Apex_IHC', 'KO_Base_IHC', 'KO_Middle_IHC', 'KO_Apex_IHC',
                  'WT_Base_OHC', 'WT_Middle_OHC', 'WT_Apex_OHC', 'KO_Base_OHC', 'KO_Middle_OHC', 'KO_Apex_OHC']

    # Use only the groups actually present (prevents NaNs and annotation errors)
    present = df['Tonotopic_KO_WT_CLass'].unique().tolist()
    order = [g for g in full_order if g in present]

    # Set up the matplotlib figure
    plt.figure(figsize=(14, 9))

    # Create a violin plot with sorted order
    violin_plot = sns.violinplot(x='Tonotopic_KO_WT_CLass', y='Distance', data=df, order=order, palette="Set3", cut=0)

    # Calculate the medians and annotate them in the violin plot
    medians = df.groupby(['Tonotopic_KO_WT_CLass'])['Distance'].median().reindex(order)
    # Vertical offset for text annotations
    vertical_offset = df['Distance'].median() * 0.05

    # Annotate medians inside the violin plots (use iloc to index by position)
    for i, xtick in enumerate(violin_plot.get_xticks()):
        y = float(medians.iloc[i])
        violin_plot.text(xtick, y + vertical_offset, f'Median: {y:.2f}',
                         horizontalalignment='center', size='medium', color='r', weight='semibold')

    # Setting labels and title
    plt.xlabel('')
    plt.ylabel('Bundle Height (\u03bcm)', fontsize=30)
    plt.xticks(rotation=90, fontsize=20)  # Rotate labels for better readability
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'bundle_height.png'), dpi=600, bbox_inches='tight', pad_inches=0.05)
    plt.show()

def plot_Normalized_total_intensities_by_tonotopic_and_group(df, output_path):
    # Create a new column combining 'Tonotopic' and 'Group' for clear labeling on the plot
    df['Tonotopic_KO_WT_CLass'] = df['KO_WT'] + '_' + df['tonotopic'].str.capitalize() + '_' + df['CLass']

    # Explicitly define the order of the groups
    full_order = ['WT_Base_IHC', 'WT_Middle_IHC', 'WT_Apex_IHC', 'KO_Base_IHC', 'KO_Middle_IHC', 'KO_Apex_IHC',
                  'WT_Base_OHC', 'WT_Middle_OHC', 'WT_Apex_OHC', 'KO_Base_OHC', 'KO_Middle_OHC', 'KO_Apex_OHC']
    present = df['Tonotopic_KO_WT_CLass'].unique().tolist()
    order = [g for g in full_order if g in present]

    # Set up the matplotlib figure
    plt.figure(figsize=(14, 9))

    # Create a violin plot with sorted order
    violin_plot = sns.violinplot(x='Tonotopic_KO_WT_CLass', y='Normalized_total_intensities',
                                 data=df, order=order, palette="Set3", cut=0)

    # Calculate the medians and annotate them in the violin plot
    medians = df.groupby(['Tonotopic_KO_WT_CLass'])['Normalized_total_intensities'].median().reindex(order)
    # Vertical offset for text annotations
    vertical_offset = df['Normalized_total_intensities'].median() * 0.05

    # Annotate medians inside the violin plots
    for i, xtick in enumerate(violin_plot.get_xticks()):
        y = float(medians.iloc[i])
        violin_plot.text(xtick, y + vertical_offset, f'Median: {y:.2f}',
                         horizontalalignment='center', size='medium', color='r', weight='semibold')

    # Setting labels and title
    plt.xlabel('')
    plt.ylabel('Normalized_total_intensities', fontsize=30)
    plt.xticks(rotation=90, fontsize=20)  # Rotate labels for better readability
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'bundle_intensity.png'), dpi=600, bbox_inches='tight', pad_inches=0.05)
    plt.show()

    # Count the number of IHCs and OHCs in each experiment
    ihc_counts = df[df['CLass'] == 'IHC'].groupby(['Tonotopic_KO_WT_CLass']).size()
    ohc_counts = df[df['CLass'] == 'OHC'].groupby(['Tonotopic_KO_WT_CLass']).size()

    # Print the results
    print("\nNumber of IHCs in each experiment:")
    print(ihc_counts.to_string())

    print("\nNumber of OHCs in each experiment:")
    print(ohc_counts.to_string())

    # Print total counts for reference
    total_ihcs = len(df[df['CLass'] == 'IHC'])
    total_ohcs = len(df[df['CLass'] == 'OHC'])

    print(f"\nTotal IHCs: {total_ihcs}")
    print(f"Total OHCs: {total_ohcs}")


def remove_outliers(df, column):
    """ Remove outliers based on 1.5*IQR from Q1 and Q3 """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def collect_intensities(df, pathtocsv_intensities, pathtocsv_distances):
    df_intensities = pd.read_csv(pathtocsv_intensities)
    df_distances = pd.read_csv(pathtocsv_distances)
    return df_intensities, df_distances


#-------------------------- pairwise statitical test

def hedges_g(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    nx, ny = len(x), len(y)
    sx2, sy2 = np.var(x, ddof=1), np.var(y, ddof=1)
    sp = np.sqrt(((nx-1)*sx2 + (ny-1)*sy2) / (nx + ny - 2))
    if sp == 0:
        return np.nan
    d = (np.mean(x) - np.mean(y)) / sp
    # small-sample bias correction
    J = 1 - (3 / (4*(nx+ny) - 9))
    return J * d

def pairwise_tonotopic_tests(
    df,
    value_col="Distance",
    group_cols=("KO_WT","CLass"),
    tonotopic_col="tonotopic",
    order=("Base","Middle","Apex"),
    include_base_vs_apex=False,
    adjust_fn=holm_adjust  # your function from above
):
    # normalize tonotopic labels to Title case for reliable matching
    d = df.copy()
    d[tonotopic_col] = d[tonotopic_col].astype(str).str.strip().str.capitalize()
    results = []

    # which pairs to test
    pairs = [("Base","Middle"), ("Middle","Apex")]
    if include_base_vs_apex:
        pairs.append(("Base","Apex"))

    # loop over strata (e.g., WT/IHC, WT/OHC, KO/IHC, KO/OHC)
    for keys, g in d.groupby(list(group_cols)):
        g = g[g[tonotopic_col].isin(order)]
        # ensure consistent order if you later need it
        for a,b in pairs:
            xa = g.loc[g[tonotopic_col]==a, value_col].dropna().values
            xb = g.loc[g[tonotopic_col]==b, value_col].dropna().values
            if len(xa) < 2 or len(xb) < 2:
                # not enough data for a t-test
                res = dict(zip(group_cols, keys))
                res.update(dict(
                    value=value_col, level1=a, level2=b,
                    n1=len(xa), n2=len(xb),
                    mean1=np.mean(xa) if len(xa) else np.nan,
                    mean2=np.mean(xb) if len(xb) else np.nan,
                    mean_diff=(np.mean(xa)-np.mean(xb)) if len(xa) and len(xb) else np.nan,
                    t=np.nan, dof=np.nan, p=np.nan, ci_low=np.nan, ci_high=np.nan, g=np.nan
                ))
                results.append(res)
                continue

            tstat, pval, dof = ttest_ind(xa, xb, usevar='unequal', alternative='two-sided')  # Welch
            cm = CompareMeans(DescrStatsW(xa), DescrStatsW(xb))
            ci_low, ci_high = cm.tconfint_diff(usevar='unequal', alternative='two-sided')
            res = dict(zip(group_cols, keys))
            res.update(dict(
                value=value_col, level1=a, level2=b,
                n1=len(xa), n2=len(xb),
                mean1=float(np.mean(xa)), mean2=float(np.mean(xb)),
                mean_diff=float(np.mean(xa)-np.mean(xb)),
                t=float(tstat), dof=float(dof), p=float(pval),
                ci_low=float(ci_low), ci_high=float(ci_high),
                g=float(hedges_g(xa, xb))
            ))
            results.append(res)

    out = pd.DataFrame(results)

    # Holm adjust p-values *within each stratum* (per value_col, per KO_WT×CLass)
    def _adj(group):
        mask = group['p'].notna()
        if mask.sum() > 0:
            group.loc[mask, 'p_holm'] = adjust_fn(group.loc[mask, 'p'].values)
        else:
            group['p_holm'] = np.nan
        return group

    out = (out
           .groupby(list(group_cols) + ['value'], group_keys=False)
           .apply(_adj)
           .sort_values(list(group_cols) + ['value','level1','level2'])
           .reset_index(drop=True))
    return out

