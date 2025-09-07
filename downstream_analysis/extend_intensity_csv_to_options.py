"""
Purpose
-------
Extend the VASCilia processed-stack output
`All_bundles_per_layer_intensity.csv` (found under
.../Intensity_response/Allcells/) into **three standardized options**
for downstream plotting/analysis (Option1, Option2, Option3).

How it works
------------
- You provide a single input: the full path to
  `All_bundles_per_layer_intensity.csv`.
- The script reads that table and generates three derived outputs
  (the “options”), each representing a different summarization/format
  of the same intensity data to suit common figure/analysis needs.
- All results are written next to the CSV in an `Intensity_plots/`
  folder (created if missing).

Input
-----
CSV_FILE_INPUT : str
    Absolute path to the `All_bundles_per_layer_intensity.csv` produced by
    the VASCilia pipeline in `.../Intensity_response/Allcells/`.

Output
------
- `Intensity_plots/option1.csv`
- `Intensity_plots/option2.csv`
- `Intensity_plots/option3.csv`
(and any accompanying figures/logs as implemented)

Usage
-----
1) Set `CSV_FILE_INPUT` to the target CSV file.
2) Run the script. The `Intensity_plots/` folder will contain the three
   derived option files.

Notes
-----
- The script assumes the CSV schema produced by VASCilia.
- No other inputs are required.

The oprions are:
- Option 1: include ALL bundles; sum all layers (zeros included).
- Option 2: ignore bundles with < 2 VALID layers; for the rest, n = smallest VALID shared depth;
            sum top-n strongest VALID layers per bundle. (Optional: placeholders.)
- Option 3: include ALL bundles; pad to n = max VALID depth by duplicating weakest VALID value.

A layer is VALID if Total Intensity > ZERO_THRESHOLD.

Also:
- Class-based coloring (IHC vs OHC1/2/3) is loaded from a separate CSV in the same folder:
  'region_intensities.csv'. If missing, default colors are used.
- Axis labels: x = "Bundle ID" (fontsize=20),
               y = "Normalized Total Intensity for Phalloidin" (fontsize=20) for normalized plots,
                   "Total Intensity for Phalloidin" for raw plots.
"""

from pathlib import Path
from typing import Optional, Tuple, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ======================== CONFIG ========================


CSV_FILE = r"C:\Users\Yasmin\Yasmin\work\My_papers\VASCilia\PLOS_review\Reviewer1\processed_data_WT_KO\WTC57BL6JMouse6MIDDLE_AiryscanProcessing\Intensity_response\Allcells\All_bundles_per_layer_intensity.csv"

OUTDIR = Path(CSV_FILE).parent / "Intensity_plots"  # -> ...\Allcells\Intensity_plots
OUTDIR.mkdir(parents=True, exist_ok=True)

# Aux file (same folder as CSV_FILE) that holds Region ID -> class labels
AUX_CLASS_FILENAME = "region_intensities.csv"   # case-insensitive match allowed

# Valid-layer rule
ZERO_THRESHOLD = 0.0          # a layer is VALID if 'Total Intensity' > ZERO_THRESHOLD

# Option 2 rule
MIN_VALID_LAYERS_OPT2 = 2     # ignore bundles with fewer than this many VALID layers
N_SHARED_OVERRIDE: Optional[int] = None   # force a specific n for Option 2 (optional)

# Option 3 rule
N_FOR_PADDING: Optional[int] = None       # if None, use max VALID depth
PAD_STRATEGY = "min"          # "min" | "repeat_last" | "zeros"

# Normalization for “_norm” plots: "max" | "sum" | "zscore" | "minmax"
NORMALIZE_MODE = "max"

# Keep identical x-axis ordering (ascending Bundle ID)
# For Option 2: put zero-height placeholders for ignored IDs?
OPT2_SHOW_PLACEHOLDERS = True

# Colors
IHC_COLOR = "#F1E05A"    # red-ish
OHC_COLOR = "#E57373"  # yellow
# ========================================================


# -------------------- I/O & utils --------------------
def load_and_clean(csv_path: str) -> pd.DataFrame:
    df_raw = pd.read_csv(csv_path)

    # Normalize column names used by the script
    col_map = {"Region ID": "region", "Z Index": "z",
               "Total Intensity": "total", "Mean Intensity": "mean"}
    df = df_raw.rename(columns=col_map).copy()

    # Types
    df["region"] = pd.to_numeric(df["region"], errors="coerce").astype("Int64")
    df["z"]      = pd.to_numeric(df["z"], errors="coerce").astype("Int64")
    df["total"]  = pd.to_numeric(df["total"], errors="coerce")

    # Drop rows with missing essentials
    df = df.dropna(subset=["region", "z", "total"]).copy()
    return df


def load_region_class_map_from_file(main_csv_path: str,
                                    aux_filename: str = AUX_CLASS_FILENAME) -> Optional[Dict[int, str]]:
    base = Path(main_csv_path).parent
    path = base / aux_filename
    if not path.exists():
        # case-insensitive fallback
        cand = None
        for p in base.glob("*.csv"):
            if p.stem.lower().startswith("region_intensities"):
                cand = p
                break
        if cand is None:
            print("No region_intensities.csv found — skipping class-based colors.")
            return None
        path = cand

    try:
        dfc = pd.read_csv(path)
    except Exception as e:
        print(f"Could not read {path.name} ({e}) — skipping class-based colors.")
        return None

    name_map = {c.lower(): c for c in dfc.columns}
    region_col = next((name_map[k] for k in ("region id","region","bundle id","id") if k in name_map), None)
    if region_col is None:
        print(f"{path.name} missing a Region ID column — skipping class-based colors.")
        return None

    class_col = next((c for c in dfc.columns if c.lower() == "class"), None)
    if class_col is None:
        print(f"{path.name} missing a 'class' column — skipping class-based colors.")
        return None

    dfc = dfc[[region_col, class_col]].dropna()
    dfc[region_col] = pd.to_numeric(dfc[region_col], errors="coerce").astype("Int64")
    dfc = dfc.dropna(subset=[region_col]).copy()
    dfc[class_col] = dfc[class_col].astype(str).str.strip()

    if dfc.empty:
        print(f"{path.name} has no usable class rows — skipping class-based colors.")
        return None

    cls_series = dfc.groupby(region_col)[class_col].agg(lambda s: s.dropna().mode().iloc[0])
    cls_map = {int(k): str(v) for k, v in cls_series.items()}
    print(f"Loaded class labels for {len(cls_map)} bundles from: {path.name}")
    return cls_map



def valid_depth_per_bundle(df: pd.DataFrame, zero_threshold: float) -> pd.Series:
    df_valid = df[df["total"] > zero_threshold]
    return df_valid.groupby("region")["z"].nunique()


def sort_by_region(df_tot: pd.DataFrame, all_regions: np.ndarray) -> pd.DataFrame:
    """
    Ensure every Region ID is present in a fixed, ascending order.
    Missing regions are inserted with NaN totals (plotted as 0).
    """
    d = df_tot.set_index("region").reindex(all_regions).reset_index()
    d = d.rename(columns={"index": "region"})
    return d


def normalize_totals(df_tot: pd.DataFrame, mode: str = "max") -> pd.DataFrame:
    d = df_tot.copy()
    x = d["total_sum"].to_numpy(dtype=float)
    mode = mode.lower()

    if mode == "max":
        denom = np.nanmax(x)
        d["total_norm"] = x / denom if denom and not np.isnan(denom) else np.zeros_like(x)
        ylab = "Total Intensity (normalized to max)"
    elif mode == "sum":
        denom = np.nansum(x)
        d["total_norm"] = x / denom if denom and not np.isnan(denom) else np.zeros_like(x)
        ylab = "Total Intensity (fraction of sum)"
    elif mode == "zscore":
        mu, sd = np.nanmean(x), np.nanstd(x, ddof=0)
        d["total_norm"] = (x - mu) / sd if sd and not np.isnan(sd) else np.zeros_like(x)
        ylab = "Total Intensity (z-score)"
    elif mode == "minmax":
        xmin, xmax = np.nanmin(x), np.nanmax(x)
        denom = xmax - xmin
        d["total_norm"] = (x - xmin) / denom if denom and not np.isnan(denom) else np.zeros_like(x)
        ylab = "Total Intensity (min–max 0–1)"
    else:
        raise ValueError("NORMALIZE_MODE must be 'max', 'sum', 'zscore', or 'minmax'.")

    d.attrs["ylab_norm"] = ylab
    return d


# -------------------- coloring + plotting --------------------
UNKNOWN_FALLBACK_COLOR = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])[0]

def make_colors_for_regions(regions: pd.Series,
                            class_map: Optional[Dict[int, str]],
                            ihc_color: str = IHC_COLOR,
                            ohc_color: str = OHC_COLOR) -> Optional[list]:
    if not class_map:
        return None
    colors = []
    for rid in regions.fillna(-1).astype(int):
        cls = str(class_map.get(rid, "")).strip().upper()
        if cls == "IHC":
            colors.append(ihc_color)
        elif cls.startswith("OHC"):  # OHC1/2/3
            colors.append(ohc_color)
        else:
            colors.append(UNKNOWN_FALLBACK_COLOR)  # safe fallback
    return colors


def barplot(df_tot: pd.DataFrame, title: str, outfile: Path,
            value_col: str = "total_sum",
            class_map: Optional[dict] = None,
            normalized: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    x_labels = df_tot["region"].astype(str)
    y = df_tot[value_col].fillna(0.0)

    colors = make_colors_for_regions(df_tot["region"], class_map)
    if colors is None:
        ax.bar(x_labels, y)
    else:
        ax.bar(x_labels, y, color=colors)

    # remove extra space and start at 0
    ax.margins(x=0)         # no left/right padding
    ax.set_ylim(bottom=0)   # y starts at 0

    # axis labels (font size 20 as requested)
    ax.set_xlabel("Bundle ID", fontsize=20)
    if normalized:
        ax.set_ylabel("Normalized Total Intensity for Phalloidin", fontsize=20)
    else:
        ax.set_ylabel("Total Intensity for Phalloidin", fontsize=20)

    ax.set_title(title)
    ax.set_xticklabels(x_labels, rotation=90)
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {outfile}")



# -------------------- Option logic --------------------
def totals_option1_all_layers(df: pd.DataFrame) -> pd.DataFrame:
    """Sum ALL layers per bundle (zeros included)."""
    return df.groupby("region", as_index=False)["total"].sum().rename(columns={"total": "total_sum"})


def totals_option2_top_n_shared_valid(
    df: pd.DataFrame,
    min_valid_layers: int = MIN_VALID_LAYERS_OPT2,
    zero_threshold: float = ZERO_THRESHOLD,
    n_override: Optional[int] = N_SHARED_OVERRIDE
) -> Tuple[pd.DataFrame, int, pd.Index]:
    """
    Ignore bundles with < min_valid_layers VALID layers.
    n = smallest VALID shared depth among eligible bundles (or override).
    For each eligible bundle, sum top-n strongest VALID layers.
    """
    df_valid = df[df["total"] > zero_threshold].copy()
    depth_valid = df_valid.groupby("region")["z"].nunique()

    eligible = depth_valid[depth_valid >= min_valid_layers].index
    if len(eligible) == 0:
        raise ValueError(f"No bundles have valid depth >= {min_valid_layers} for Option 2.")

    n_candidate = int(depth_valid.loc[eligible].min())
    n = min(n_override, n_candidate) if n_override else n_candidate
    print(f"[Option 2] min_valid_layers={min_valid_layers} → n={n} "
          f"(min shared VALID depth among eligible={n_candidate}; override={n_override})")
    print(f"[Option 2] Eligible: {len(eligible)} | Ignored (<{min_valid_layers} valid): "
          f"{df['region'].nunique() - len(eligible)}")

    rows = []
    for rid, dfg in df_valid[df_valid["region"].isin(eligible)].groupby("region"):
        strongest = dfg["total"].nlargest(n).to_numpy()
        rows.append({"region": int(rid), "total_sum": float(strongest.sum())})

    return pd.DataFrame(rows), n, eligible


def totals_option3_pad_to_n_valid(
    df: pd.DataFrame,
    n: Optional[int] = N_FOR_PADDING,
    pad_strategy: str = PAD_STRATEGY,
    zero_threshold: float = ZERO_THRESHOLD
) -> Tuple[pd.DataFrame, int]:
    """
    Include ALL bundles. n = max VALID depth (or override).
    For each bundle, keep VALID layers; if fewer than n, pad by duplicating its weakest VALID value.
    """
    df_valid = df[df["total"] > zero_threshold].copy()
    depth_valid = df_valid.groupby("region")["z"].nunique()

    n_pad = int(depth_valid.max()) if n is None else int(n)
    print(f"[Option 3] Padding to n = {n_pad} (max VALID depth = {int(depth_valid.max())})")

    rows = []
    for rid in sorted(df["region"].unique()):
        dfg_valid = df_valid[df_valid["region"] == rid].sort_values("z")
        vals = dfg_valid["total"].to_numpy()

        if len(vals) == 0:
            total_sum = 0.0
        else:
            if len(vals) < n_pad:
                if pad_strategy == "min":
                    pad_val = float(vals.min())
                    pad = [pad_val] * (n_pad - len(vals))
                elif pad_strategy == "repeat_last":
                    pad = [float(vals[-1])] * (n_pad - len(vals))
                else:  # zeros
                    pad = [0.0] * (n_pad - len(vals))
                vals = np.concatenate([vals, pad], axis=0)
            elif len(vals) > n_pad:
                vals = vals[:n_pad]
            total_sum = float(vals.sum())

        rows.append({"region": int(rid), "total_sum": total_sum})

    return pd.DataFrame(rows), n_pad


# -------------------- Main --------------------
def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df = load_and_clean(CSV_FILE)

    # Build class color map from aux file
    class_map = load_region_class_map_from_file(CSV_FILE, AUX_CLASS_FILENAME)
    if class_map:
        print(f"Coloring IHC={IHC_COLOR}, OHC*={OHC_COLOR}.")
    else:
        print("No class map available — default colors will be used.")

    all_regions = np.array(sorted(df["region"].unique()))
    print(f"Bundles: {len(all_regions)}")

    # Diagnostics
    depth_raw   = df.groupby("region")["z"].nunique()
    depth_valid = valid_depth_per_bundle(df, ZERO_THRESHOLD)
    print("Raw depths (counting zeros):", sorted(depth_raw.unique()))
    print("VALID depths (> threshold): ", sorted(depth_valid.unique()))
    if class_map:
        missing = sorted(set(int(r) for r in all_regions) - set(class_map.keys()))
        if missing:
            print("Bundles without class label in aux file (colored with fallback):", missing)

    # ---------- Option 1 ----------
    opt1 = totals_option1_all_layers(df)
    opt1 = sort_by_region(opt1, all_regions)  # include ALL IDs in fixed order
    opt1.to_csv(OUTDIR / "option1_totals.csv", index=False)
    barplot(opt1, "Option 1 — Total Intensity (All layers; zeros included)",
            OUTDIR / "option1_totals.png",
            value_col="total_sum", class_map=class_map, normalized=False)

    opt1_norm = normalize_totals(opt1, NORMALIZE_MODE)
    opt1_norm.to_csv(OUTDIR / f"option1_totals_norm_{NORMALIZE_MODE}.csv", index=False)
    barplot(opt1_norm, f"Option 1 — Normalized ({NORMALIZE_MODE})",
            OUTDIR / f"option1_totals_norm_{NORMALIZE_MODE}.png",
            value_col="total_norm", class_map=class_map, normalized=True)

    # ---------- Option 2 ----------
    opt2, n_shared, eligible = totals_option2_top_n_shared_valid(
        df,
        min_valid_layers=MIN_VALID_LAYERS_OPT2,
        zero_threshold=ZERO_THRESHOLD,
        n_override=N_SHARED_OVERRIDE
    )
    if OPT2_SHOW_PLACEHOLDERS:
        opt2 = sort_by_region(opt2, all_regions)  # zero-height bars for ignored IDs
    else:
        opt2 = opt2.sort_values("region")

    opt2.to_csv(OUTDIR / f"option2.csv", index=False)
    barplot(
        opt2,
        f"Option 2 — Total Intensity (Top {n_shared} strongest VALID; ignore < {MIN_VALID_LAYERS_OPT2} valid)",
        OUTDIR / f"option2_totals_top{n_shared}_valid.png",
        value_col="total_sum", class_map=class_map, normalized=False
    )

    opt2_norm = normalize_totals(opt2, NORMALIZE_MODE)
    opt2_norm.to_csv(OUTDIR / f"option2_totals_top{n_shared}_valid_norm_{NORMALIZE_MODE}.csv", index=False)
    barplot(
        opt2_norm,
        f"Option 2 — Normalized ({NORMALIZE_MODE})",
        OUTDIR / f"option2_totals_top{n_shared}_valid_norm_{NORMALIZE_MODE}.png",
        value_col="total_norm", class_map=class_map, normalized=True
    )

    # ---------- Option 3 ----------
    opt3, n_pad = totals_option3_pad_to_n_valid(
        df,
        n=N_FOR_PADDING,
        pad_strategy=PAD_STRATEGY,
        zero_threshold=ZERO_THRESHOLD
    )
    opt3 = sort_by_region(opt3, all_regions)  # include ALL IDs in fixed order
    opt3.to_csv(OUTDIR / f"option3.csv", index=False)
    barplot(
        opt3,
        f"Option 3 — Total Intensity (Padded to n={n_pad} VALID; weakest duplicated)",
        OUTDIR / f"option3_totals_pad{n_pad}_valid.png",
        value_col="total_sum", class_map=class_map, normalized=False
    )

    opt3_norm = normalize_totals(opt3, NORMALIZE_MODE)
    opt3_norm.to_csv(OUTDIR / f"option3_totals_pad{n_pad}_valid_norm_{NORMALIZE_MODE}.csv", index=False)
    barplot(
        opt3_norm,
        f"Option 3 — Normalized ({NORMALIZE_MODE})",
        OUTDIR / f"option3_totals_pad{n_pad}_valid_norm_{NORMALIZE_MODE}.png",
        value_col="total_norm", class_map=class_map, normalized=True
    )

    print("Done.")


if __name__ == "__main__":
    main()
