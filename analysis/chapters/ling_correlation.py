import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import cosine

# Mapping 2-letter codes to ISO 639-3 codes for URIEL/lang2vec
ISO_MAP = {
    "ar": "ara",
    "bn": "ben",
    "de": "deu",
    "en": "eng",
    "es": "spa",
    "fr": "fra",
    "hi": "hin",
    "id": "ind",
    "it": "ita",
    "pt": "por",
    "zh": "cmn",
}


def get_linguistic_distances(src_lang, tgt_lang, features):
    """
    Computes Cosine Distance between two languages for a set of linguistic features.
    """
    src_iso = ISO_MAP.get(src_lang)
    tgt_iso = ISO_MAP.get(tgt_lang)

    if not src_iso or not tgt_iso:
        return {f: np.nan for f in features}

    dists = {}
    for ft_name, ft_dict in features.items():
        # lang2vec returns a dict like {'eng': [vector], ...}
        if src_iso in ft_dict and tgt_iso in ft_dict:
            # Cosine distance: 0 = identical, 1 = orthogonal
            # We use cosine from scipy (which handles 1-D arrays correctly)
            d = cosine(ft_dict[src_iso], ft_dict[tgt_iso])
            dists[ft_name] = d
        else:
            dists[ft_name] = np.nan
    return dists


def analyze_linguistic_correlations(
    df_transfer: pd.DataFrame,
    delta_col: str = "delta_score_norm_mean",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Correlates Transfer Performance (Delta) with Linguistic Distance.
    """

    # --- 1. Filter for Matched-Task Cross-Language (MT-CL) ---
    mask_mt_cl = (df_transfer["fine_tuned_dataset"] == df_transfer["eval_dataset"]) & (
        df_transfer["fine_tuned_language"] != df_transfer["eval_language"]
    )
    df_mt = df_transfer[mask_mt_cl].copy()

    if df_mt.empty:
        print("Warning: No Matched-Task Cross-Language rows found.")
        return pd.DataFrame()

    # --- 2. Aggregate Score per Language Pair ---
    df_pairs = (
        df_mt.groupby(["fine_tuned_language", "eval_language"])[delta_col]
        .mean()
        .reset_index()
    )

    # --- 3. Fetch Linguistic Features (FIXED) ---
    feature_types = ["syntax_knn", "phonology_knn", "inventory_knn"]

    unique_langs = list(set(ISO_MAP.values()))
    loaded_features = {}

    print("Loading lang2vec features...")
    try:
        import lang2vec.lang2vec as l2v

        for ft in feature_types:
            # Languages first, then feature name.
            # Also ensures unique_langs is a list of strings.
            loaded_features[ft] = l2v.get_features(unique_langs, ft)

    except Exception as e:
        print(f"Error loading lang2vec features: {e}")
        print(
            "Tip: You may need to run 'import lang2vec.lang2vec as l2v; l2v.download_all()' once."
        )
        return pd.DataFrame()

    # --- 4. Compute Distances ---
    dist_data = []

    for idx, row in df_pairs.iterrows():
        src = row["fine_tuned_language"]
        tgt = row["eval_language"]
        transfer_score = row[delta_col]

        dists = get_linguistic_distances(src, tgt, loaded_features)

        entry = {
            "src": src,
            "tgt": tgt,
            "transfer": transfer_score,
            "dist_syntax": dists.get("syntax_knn"),
            "dist_phonology": dists.get("phonology_knn"),
            "dist_inventory": dists.get("inventory_knn"),
        }
        dist_data.append(entry)

    df_dist = pd.DataFrame(dist_data).dropna()

    # --- 5. Compute Correlations ---
    results = []
    metrics = {
        "Syntax Distance": "dist_syntax",
        "Phonology Distance": "dist_phonology",
        "Inventory Distance": "dist_inventory",
    }

    for metric_name, col_name in metrics.items():
        if col_name not in df_dist.columns:
            continue

        x = df_dist[col_name]
        y = df_dist["transfer"]

        r, p_r = pearsonr(x, y)
        rho, p_rho = spearmanr(x, y)

        results.append(
            {
                "Metric": metric_name,
                "Spearman (Rank)": round(rho, 3),
                "p-value": round(p_rho, 4),
                "Pearson (Linear)": round(r, 3),
                "N": len(df_dist),
            }
        )

    df_results = pd.DataFrame(results)
    return df_results


def analyze_family_correlations(df_transfer, delta_col="delta_score_norm_mean"):
    """
    Fallback method using hardcoded linguistic family distances.
    Distance = 0 (Same), 0.2 (Same Cluster), 0.5 (Same Family), 1.0 (Different)
    """
    # Define Clusters
    clusters = {
        "Romance": ["es", "fr", "it", "pt"],
        "Germanic": ["en", "de"],
        "Indo-Aryan": ["hi", "bn"],
        "Semitic": ["ar"],
        "Austronesian": ["id"],
        "Sino-Tibetan": ["zh"],
    }
    # Define Super-Families
    families = {
        "Indo-European": ["es", "fr", "it", "pt", "en", "de", "hi", "bn"],
        "Afro-Asiatic": ["ar"],
        "Austronesian": ["id"],
        "Sino-Tibetan": ["zh"],
    }

    def get_family_dist(l1, l2):
        if l1 == l2:
            return 0.0
        # Check Cluster (e.g. Spanish-French)
        for c, langs in clusters.items():
            if l1 in langs and l2 in langs:
                return 0.2
        # Check Family (e.g. English-Hindi)
        for f, langs in families.items():
            if l1 in langs and l2 in langs:
                return 0.5
        # Different Family
        return 1.0

    mask_mt_cl = (df_transfer["fine_tuned_dataset"] == df_transfer["eval_dataset"]) & (
        df_transfer["fine_tuned_language"] != df_transfer["eval_language"]
    )
    df_mt = df_transfer[mask_mt_cl].copy()
    df_pairs = (
        df_mt.groupby(["fine_tuned_language", "eval_language"])[delta_col]
        .mean()
        .reset_index()
    )

    dist_data = []
    for _, row in df_pairs.iterrows():
        d = get_family_dist(row["fine_tuned_language"], row["eval_language"])
        dist_data.append({"dist_family": d, "transfer": row[delta_col]})

    df_dist = pd.DataFrame(dist_data)
    r, _ = pearsonr(df_dist["dist_family"], df_dist["transfer"])
    rho, p = spearmanr(df_dist["dist_family"], df_dist["transfer"])

    return pd.DataFrame(
        [
            {
                "Metric": "Family Distance (Hardcoded)",
                "Spearman (Rank)": round(rho, 3),
                "p-value": round(p, 4),
                "Pearson": round(r, 3),
            }
        ]
    )
