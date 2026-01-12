import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import json

# ------------------------
# Load Lahman database
# ------------------------
con = sqlite3.connect("lahman.db")

batting = pd.read_sql("SELECT * FROM Batting", con)
pitching = pd.read_sql("SELECT * FROM Pitching", con)
fielding = pd.read_sql("SELECT * FROM Fielding", con)
fieldingOF = pd.read_sql("SELECT * FROM fieldingOFSplit", con)
awards = pd.read_sql("SELECT * FROM AwardsSharePlayers", con)
teams = pd.read_sql("SELECT * FROM Teams", con)

# ------------------------
# MVP shares
# ------------------------
mvpshares = (
    awards[awards["awardID"] == "Most Valuable Player"]
    .assign(share=lambda x: x.pointsWon / x.pointsMax)
    [["yearID", "lgID", "playerID", "share"]]
)

# ------------------------
# Team win %
# ------------------------
teams_lg = (
    teams.assign(winP=lambda x: x.W / x.G)
    [["yearID", "lgID", "teamID", "winP", "LgWin"]]
)

# ------------------------
# Fielding (primary position)
# ------------------------
fielding_all = pd.concat([fielding, fieldingOF], ignore_index=True)

specific_of = (
    fielding_all[fielding_all.POS.isin(["LF", "CF", "RF"])]
    [["playerID", "yearID"]]
    .drop_duplicates()
    .assign(has_specific_of=True)
)

fielding_clean = (
    fielding_all
    .merge(specific_of, on=["playerID", "yearID"], how="left")
    .query("~(POS == 'OF' and has_specific_of == True)")
)

fielding_pos = (
    fielding_clean
    .groupby(["playerID", "yearID", "POS"], as_index=False)
    .agg(InnOuts=("InnOuts", "sum"))
    .assign(innings=lambda x: x.InnOuts / 3)
)

primary_pos = (
    fielding_pos
    .sort_values("innings", ascending=False)
    .groupby(["playerID", "yearID"])
    .head(1)[["playerID", "yearID", "POS"]]
    .rename(columns={"POS": "primary_pos"})
)

primary_pos = primary_pos.dropna(subset=["primary_pos"])
# ------------------------
# Batting / pitching cleanup
# ------------------------
# safe column accessor that returns a Series (zeros if missing)
def _col_series(df, *names):
    for n in names:
        if n in df.columns:
            return df[n].fillna(0)
    return pd.Series(0, index=df.index, dtype=float)

# compute singles (X1B) robustly even if DB uses '2B'/'3B' or 'X2B'/'X3B'
batting["X1B"] = _col_series(batting, "H") - _col_series(batting, "2B", "X2B") - _col_series(batting, "3B", "X3B") - _col_series(batting, "HR")

# FO = forced outs (approx) and BB+HBP combined
batting["FO"] = _col_series(batting, "AB") - _col_series(batting, "H") - _col_series(batting, "SO")
batting["BBHBP"] = _col_series(batting, "BB") + _col_series(batting, "HBP")

batting_less = batting.drop(
    columns=["G","AB","BB","IBB","HBP","H","SH","SF","GIDP","G_batting","G_old"],
    errors="ignore"
)

pitching["IP"] = pitching.IPouts / 3
pitching = pitching.rename(columns={"SO": "K"})

pitching_less = pitching.drop(
    columns=["IPouts","G","GS","CG","H","BB","HR","BAOpp","ERA",
             "IBB","WP","HBP","BK","BFP","GF","GIDP","SF","SH","R"],
    errors="ignore"
)

# ------------------------
# Merge batting + pitching
# ------------------------
player = pd.merge(
    batting_less,
    pitching_less,
    on=["playerID","yearID","lgID","teamID","stint"],
    how="outer"
).fillna(0)

# ------------------------
# Multi-team seasons
# ------------------------
team_counts = (
    player[["playerID","yearID","lgID","teamID"]]
    .drop_duplicates()
    .groupby(["playerID","yearID","lgID"])
    .size()
    .reset_index(name="n_teams")
)

player = player.merge(team_counts, on=["playerID","yearID","lgID"], how="left")

player["teamID"] = np.where(
    player.n_teams > 1,
    player.n_teams.astype(str) + "TM",
    player.teamID
)

player = (
    player
    .groupby(["playerID","yearID","lgID","teamID"], as_index=False)
    .sum(numeric_only=True)
)

# ------------------------
# Final merge
# ------------------------
df = (
    player
    .merge(primary_pos, on=["playerID","yearID"], how="left")
    .merge(mvpshares, on=["playerID","yearID","lgID"], how="left")
    .merge(teams_lg, on=["yearID","lgID","teamID"], how="left")
)

df["winP"] = np.where(df.teamID.str.endswith("TM"), 0.5, df.winP)
df["LgWin"] = np.where(df.teamID.str.endswith("TM"), 0, (df.LgWin == "Y").astype(int))
df["share"] = df.share.fillna(0)
df["primary_pos"] = df["primary_pos"].fillna("OF")

# ------------------------
# Lock position categories (CRITICAL)
# ------------------------
pos_levels = ["1B","2B","3B","SS","LF","CF","RF","OF","C","P"]
df["primary_pos"] = pd.Categorical(df.primary_pos, categories=pos_levels)

df = pd.get_dummies(df, columns=["primary_pos"])

# ------------------------
# Group IDs
# ------------------------
df = df.sort_values(["yearID","lgID"])
df["group_id"] = df.groupby(["yearID","lgID"]).ngroup()

# ------------------------
# Save base training set
# ------------------------
features = df.drop(
    columns=["playerID","teamID","yearID","lgID","group_id","share","stint","n_teams"]
)

pickle.dump(df, open("base_training_df.pkl", "wb"))
json.dump(list(features.columns), open("feature_columns.json", "w"))

print("Training data prepared and saved.")
print(f"Features list: {list(features.columns)}")