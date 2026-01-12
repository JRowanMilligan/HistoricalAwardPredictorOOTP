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

# We only need pitching, awards, and teams for Cy Young
pitching = pd.read_sql("SELECT * FROM Pitching", con)
awards = pd.read_sql("SELECT * FROM AwardsSharePlayers", con)
teams = pd.read_sql("SELECT * FROM Teams", con)

# ------------------------
# Cy Young shares
# ------------------------
# Filter for "Cy Young Award"
cy_shares = (
    awards[awards["awardID"] == "Cy Young Award"]
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
# Pitching cleanup
# ------------------------
# 1. Convert IPouts to IP
pitching["IP"] = pitching.IPouts / 3

# 2. Rename SO to K (to avoid batting SO conflicts)
pitching = pitching.rename(columns={"SO": "K"})

# 3. Drop unwanted columns
# We kept: W, L, G, SV, IP, K, SHO
# We drop: ERA (calculated implicitly by model if needed), H, BB, etc.
pitching_less = pitching[
    ["playerID", "yearID", "lgID", "teamID", "stint", "W", "L", "SV", "IP", "K", "SHO"]
].copy()

# ------------------------
# Multi-team seasons (Pitching specific)
# ------------------------
# If a pitcher played for multiple teams, we sum their stats
team_counts = (
    pitching_less[["playerID","yearID","lgID","teamID"]]
    .drop_duplicates()
    .groupby(["playerID","yearID","lgID"])
    .size()
    .reset_index(name="n_teams")
)

# Merge counts back
player = pitching_less.merge(team_counts, on=["playerID","yearID","lgID"], how="left")

# Create generic teamID for multi-team players
player["teamID"] = np.where(
    player.n_teams > 1,
    player.n_teams.astype(str) + "TM",
    player.teamID
)

# Sum stats for multi-team entries
player = (
    player
    .groupby(["playerID","yearID","lgID","teamID"], as_index=False)
    .sum(numeric_only=True)
)

# ------------------------
# Final merge
# ------------------------
# Note: For Cy Young, we merge on playerID and yearID primarily.
# The lgID in 'awards' might be 'ML' for 1956-1966.
# The lgID in 'player' (pitching stats) will be 'AL' or 'NL'.

# To handle the merge correctly:
# 1. Merge Pitching Stats + Team Stats first
df = player.merge(teams_lg, on=["yearID","lgID","teamID"], how="left")

# Handle Team Stats for "XTM" players
df["winP"] = np.where(df.teamID.str.endswith("TM"), 0.5, df["winP"])
df["LgWin"] = np.where(df.teamID.str.endswith("TM"), 0, (df["LgWin"] == "Y").astype(int))

# 2. Merge Award Shares
# We do a LEFT JOIN. 
# However, for 1956-1966, the award table has lgID="ML".
# The pitching table has lgID="AL" or "NL".
# We must allow the merge to find the player regardless of league label for those years.

# Strategy: Separate merge for Normal Era vs ML Era

# A. Normal Era (Not 1956-1966)
df_normal = df[~((df.yearID >= 1956) & (df.yearID <= 1966))].copy()
cy_normal = cy_shares[~((cy_shares.yearID >= 1956) & (cy_shares.yearID <= 1966))].copy()

df_normal = df_normal.merge(cy_normal, on=["playerID", "yearID", "lgID"], how="left")

# B. ML Era (1956-1966)
df_ml_era = df[(df.yearID >= 1956) & (df.yearID <= 1966)].copy()
cy_ml_era = cy_shares[(cy_shares.yearID >= 1956) & (cy_shares.yearID <= 1966)].copy()

# Drop lgID from the AWARD table merge keys so we match purely on Player + Year
df_ml_era = df_ml_era.merge(cy_ml_era.drop(columns=["lgID"]), on=["playerID", "yearID"], how="left")

# Combine back
df_final = pd.concat([df_normal, df_ml_era], ignore_index=True)

# Fill missing shares with 0
df_final["share"] = df_final["share"].fillna(0)

# ------------------------
# Group IDs (The Ranking Pools)
# ------------------------
# For 1956-1966, the competition pool is "Year_ML" (everyone in MLB).
# For all other years, the competition pool is "Year_League".

def get_group_id(row):
    if 1956 <= row["yearID"] <= 1966:
        return f"{row['yearID']}_ML"
    else:
        return f"{row['yearID']}_{row['lgID']}"

df_final["group_id"] = df_final.apply(get_group_id, axis=1)

# Clean up structural columns not needed for training
# We removed: ERA, WSWin.
# We keep: W, L, G, SV, IP, K, SHO, winP, LgWin
features_to_drop = [
    "playerID", "teamID", "yearID", "lgID", 
    "group_id", "share", "stint", "n_teams"
]

# Ensure we aren't dropping something that doesn't exist
features_to_drop = [c for c in features_to_drop if c in df_final.columns]

features = df_final.drop(columns=features_to_drop)

# ------------------------
# Save Cy Young training set
# ------------------------
pickle.dump(df_final, open("cy_training_df.pkl", "wb"))
json.dump(list(features.columns), open("cy_features.json", "w"))

print("Cy Young training data prepared and saved.")
print(f"Features list: {list(features.columns)}")