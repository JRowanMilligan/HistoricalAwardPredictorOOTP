"""Evaluate the award rankers with season-isolated, local-history holdouts.

For every eligible target season, this script trains a fresh ranker using only
the seasons immediately before and after it.  The target season is never in
the training set, so the resulting scores answer the useful question: given
the voting patterns around a season, how well would the model reproduce that
season's ballot?

Examples
--------
python evaluate_award_models.py
python evaluate_award_models.py --award mvp --window 5 --boost-rounds 300
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "graphs" / "model_evaluation"

AWARDS = {
    "mvp": {
        "label": "MVP",
        "data": ROOT / "base_training_df.pkl",
        "features": ROOT / "feature_columns.json",
        "depth": 6,
        "default_rounds": 300,
    },
    "cy": {
        "label": "Cy Young",
        "data": ROOT / "cy_training_df.pkl",
        "features": ROOT / "cy_features.json",
        "depth": 4,
        "default_rounds": 200,
    },
}


def rank_descending(values: pd.Series, method: str = "first") -> pd.Series:
    """Rank higher values first; average ranks keep real vote-share ties intact."""
    return values.rank(method=method, ascending=False)


def ndcg_at_k(actual: np.ndarray, predicted: np.ndarray, k: int = 3) -> float:
    """NDCG with vote share as relevance; 1.0 is a perfect ballot ordering."""
    order = np.argsort(-predicted)[:k]
    ideal = np.argsort(-actual)[:k]
    discounts = 1.0 / np.log2(np.arange(2, len(order) + 2))
    dcg = np.sum((2.0**actual[order] - 1.0) * discounts)
    idcg = np.sum((2.0**actual[ideal] - 1.0) * discounts)
    return float(dcg / idcg) if idcg > 0 else np.nan


def spearman_from_ranks(left: pd.Series, right: pd.Series) -> float:
    """Spearman correlation without making scipy a project requirement."""
    if len(left) < 2 or left.nunique() < 2 or right.nunique() < 2:
        return np.nan
    return float(left.corr(right, method="pearson"))


def make_training_matrix(frame: pd.DataFrame, features: list[str]):
    ordered = frame.sort_values(["group_id", "yearID", "playerID"]).reset_index(drop=True)
    sizes = ordered.groupby("group_id", sort=False).size().tolist()
    matrix = xgb.DMatrix(ordered[features].astype(float), label=ordered["share"].astype(float))
    matrix.set_group(sizes)
    return matrix


def score_target(
    model: xgb.Booster, target: pd.DataFrame, features: list[str], year: int
) -> list[dict]:
    scored = target.copy()
    scored["prediction"] = model.predict(xgb.DMatrix(scored[features].astype(float)))
    rows = []
    # A year can contain one MLB-wide pool or separate AL/NL pools.  Keep both
    # in the raw output, then aggregate to one season-level result.
    for pool, ballot in scored.groupby("group_id", sort=False):
        ballot = ballot.copy()
        # Seasons before an award existed have no ballot at all: every share is
        # zero.  They are useful prediction rows, but cannot validate voting.
        if ballot["share"].max() <= 0:
            continue
        # Most candidates received no votes.  Treat that as an actual tie for
        # correlation rather than inventing an order among zero-share players.
        ballot["actual_rank"] = rank_descending(ballot["share"], method="average")
        ballot["predicted_rank"] = rank_descending(ballot["prediction"], method="first")
        winner = ballot.loc[ballot["share"].idxmax()]
        winner_rank = int(winner["predicted_rank"])
        rows.append(
            {
                "year": year,
                "pool": pool,
                "players": len(ballot),
                "ndcg_at_3": ndcg_at_k(
                    ballot["share"].to_numpy(), ballot["prediction"].to_numpy(), k=3
                ),
                "spearman": spearman_from_ranks(
                    ballot["actual_rank"], ballot["predicted_rank"]
                ),
                "winner_rank": winner_rank,
                "winner_top_1": int(winner_rank == 1),
                "winner_top_3": int(winner_rank <= 3),
                "winner_mrr": 1.0 / winner_rank,
            }
        )
    return rows


def evaluate_award(key: str, window: int, rounds: int, seed: int) -> pd.DataFrame:
    config = AWARDS[key]
    frame = pickle.load(open(config["data"], "rb"))
    features = json.load(open(config["features"]))
    years = sorted(frame["yearID"].unique())
    results = []

    print(f"Evaluating {config['label']} ({len(years)} seasons; +/- {window} years)...")
    for index, year in enumerate(years, start=1):
        target = frame.loc[frame["yearID"] == year]
        # Do not spend a training run on years before the award had a ballot.
        if target["share"].max() <= 0:
            continue
        train = frame.loc[
            (frame["yearID"].between(year - window, year + window))
            & (frame["yearID"] != year)
        ]
        # A ranker cannot train without at least two historical ballot pools.
        if train["group_id"].nunique() < 2 or target["group_id"].empty:
            continue

        params = {
            "objective": "rank:pairwise",
            "eval_metric": "ndcg",
            "eta": 0.05,
            "max_depth": config["depth"],
            "seed": seed,
            "tree_method": "hist",
            "nthread": 4,
        }
        model = xgb.train(params, make_training_matrix(train, features), num_boost_round=rounds)
        results.extend(score_target(model, target, features, int(year)))
        if index % 10 == 0 or index == len(years):
            print(f"  {index}/{len(years)} target seasons complete")

    return pd.DataFrame(results)


def plot_results(per_pool: pd.DataFrame, award: str, window: int) -> pd.DataFrame:
    """Write a season trend and true-winner-rank graphic, plus CSV data."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    season = (
        per_pool.groupby("year", as_index=False)
        .agg(
            ndcg_at_3=("ndcg_at_3", "mean"),
            spearman=("spearman", "mean"),
            winner_top_1=("winner_top_1", "mean"),
            winner_top_3=("winner_top_3", "mean"),
            winner_mrr=("winner_mrr", "mean"),
            pools=("pool", "count"),
        )
        .sort_values("year")
    )
    slug = award.lower().replace(" ", "_")
    per_pool.to_csv(OUTPUT_DIR / f"{slug}_pool_results.csv", index=False)
    season.to_csv(OUTPUT_DIR / f"{slug}_season_results.csv", index=False)

    rolling = season.set_index("year")[["ndcg_at_3", "winner_top_1", "winner_top_3"]].rolling(
        9, center=True, min_periods=3
    ).mean()
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    fig.suptitle(f"{award} local-history holdout performance (±{window} seasons)", fontsize=15, weight="bold")
    axes[0].plot(season["year"], season["ndcg_at_3"], color="#9ecae1", alpha=0.45, label="Season score")
    axes[0].plot(rolling.index, rolling["ndcg_at_3"], color="#08519c", linewidth=2.5, label="9-season average")
    axes[0].set_ylabel("NDCG@3 (vote-share ranking)")
    axes[0].set_ylim(0, 1.03)
    axes[0].legend(frameon=False, loc="lower right")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].plot(rolling.index, rolling["winner_top_1"], color="#cb181d", linewidth=2.5, label="Correct winner")
    axes[1].plot(rolling.index, rolling["winner_top_3"], color="#238b45", linewidth=2.5, label="Winner in top 3")
    axes[1].set_ylabel("Share of ballot pools")
    axes[1].set_xlabel("Award season")
    axes[1].set_ylim(0, 1.03)
    axes[1].legend(frameon=False, loc="lower right")
    axes[1].grid(axis="y", alpha=0.25)
    fig.savefig(OUTPUT_DIR / f"{slug}_seasonal_performance.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    ranks = per_pool["winner_rank"].clip(upper=10)
    labels = [str(i) for i in range(1, 10)] + ["10+"]
    counts = [int((ranks == i).sum()) for i in range(1, 10)] + [int((ranks >= 10).sum())]
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    bars = ax.bar(labels, counts, color=["#cb181d"] + ["#6baed6"] * 8 + ["#bdbdbd"])
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_title(f"Where the actual {award} winner ranked in isolated testing", weight="bold")
    ax.set_xlabel("Model rank for the actual winner")
    ax.set_ylabel("Number of award ballot pools")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(OUTPUT_DIR / f"{slug}_winner_rank_distribution.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    return season


def write_summary(all_seasons: dict[str, pd.DataFrame], window: int) -> None:
    summary = []
    for award, season in all_seasons.items():
        summary.append(
            {
                "award": award,
                "seasons": len(season),
                "mean_ndcg_at_3": season["ndcg_at_3"].mean(),
                "mean_spearman": season["spearman"].mean(),
                "winner_top_1_rate": season["winner_top_1"].mean(),
                "winner_top_3_rate": season["winner_top_3"].mean(),
                "mean_winner_mrr": season["winner_mrr"].mean(),
                "window_years": window,
            }
        )
    pd.DataFrame(summary).to_csv(OUTPUT_DIR / "summary.csv", index=False)
    with open(OUTPUT_DIR / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--award", choices=["mvp", "cy", "all"], default="all")
    parser.add_argument("--window", type=int, default=5, help="Seasons before and after each holdout")
    parser.add_argument("--boost-rounds", type=int, help="Override the matching app model's tree count")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.window < 1:
        parser.error("--window must be at least 1")

    keys = list(AWARDS) if args.award == "all" else [args.award]
    all_seasons = {}
    for key in keys:
        rounds = args.boost_rounds or AWARDS[key]["default_rounds"]
        per_pool = evaluate_award(key, args.window, rounds, args.seed)
        if per_pool.empty:
            raise RuntimeError(f"No eligible {AWARDS[key]['label']} holdouts were produced.")
        all_seasons[AWARDS[key]["label"]] = plot_results(per_pool, AWARDS[key]["label"], args.window)
    write_summary(all_seasons, args.window)
    print(f"Results and graphics written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
