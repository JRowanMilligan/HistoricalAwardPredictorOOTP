import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import pickle
import json
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="Historical Award Predictor for OOTP", layout="wide")
st.title("Historical Baseball Award Predictor for OOTP")

# ------------------------
# Load Models & Features
# ------------------------
@st.cache_resource
def load_models():
    models = {}
    features = {}
    
    # 1. MVP Model
    if os.path.exists("base_training_df.pkl") and os.path.exists("feature_columns.json"):
        if "mvp_model" in st.session_state:
            models["MVP"] = st.session_state["mvp_model"]
            features["MVP"] = json.load(open("feature_columns.json"))
        elif os.path.exists("model.pkl"): 
             models["MVP"] = pickle.load(open("model.pkl", "rb"))
             features["MVP"] = json.load(open("feature_columns.json"))

    # 2. Cy Young Model
    if os.path.exists("cy_model.pkl") and os.path.exists("cy_features.json"):
        models["CY"] = pickle.load(open("cy_model.pkl", "rb"))
        features["CY"] = json.load(open("cy_features.json"))
    
    return models, features

# Initialize session state for MVP training
if "mvp_model" not in st.session_state and os.path.exists("base_training_df.pkl"):
    pass

# ------------------------
# User Inputs (Training Side)
# ------------------------
with st.expander("MVP Model Training (Open to Retrain)"):
    target_year = st.number_input("Era center year", 1871, 2025, 1977)
    
    if st.button("Train / Retrain MVP Model"):
        try:
            df = pickle.load(open("base_training_df.pkl", "rb"))
            feats = json.load(open("feature_columns.json"))
            
            # Sort for ranking
            df_sorted = df.sort_values(["group_id", "yearID", "lgID", "teamID", "playerID"]).reset_index(drop=True)
            X = df_sorted[feats].astype(float).values
            y = df_sorted["share"].astype(float).values
            
            group_keys = list(df_sorted["group_id"].unique())
            group_sizes = [int((df_sorted["group_id"] == g).sum()) for g in group_keys]
            
            # Recency weights
            group_weight_map = (
                df_sorted[["group_id","yearID"]]
                .drop_duplicates(subset=["group_id"])
                .set_index("group_id")
                .assign(weight=lambda x: np.exp(-abs(x.yearID - target_year) / 5))["weight"]
            )
            group_weights = [float(group_weight_map.loc[g]) for g in group_keys]

            dtrain = xgb.DMatrix(X, label=y, weight=group_weights)
            dtrain.set_group(group_sizes)
            
            params = {
                "objective": "rank:pairwise", "eval_metric": "ndcg", 
                "eta": 0.05, "max_depth": 6
            }
            
            mvp_model = xgb.train(params, dtrain, num_boost_round=300)
            st.session_state["mvp_model"] = mvp_model
            st.success("MVP Model trained!")
        except Exception as e:
            st.error(f"Training failed: {e}")

# ------------------------
# Train Cy Young Model
# ------------------------
with st.expander("Cy Young Model Training"):
    if st.button("Train / Retrain Cy Young Model"):
        try:
            cy_df = pickle.load(open("cy_training_df.pkl", "rb"))
            cy_feats = json.load(open("cy_features.json"))
            
            cy_sorted = cy_df.sort_values(["group_id", "share"], ascending=[True, False]).reset_index(drop=True)
            X_cy = cy_sorted[cy_feats].astype(float).values
            y_cy = cy_sorted["share"].astype(float).values
            
            cy_group_keys = list(cy_sorted["group_id"].unique())
            cy_group_sizes = [int((cy_sorted["group_id"] == g).sum()) for g in cy_group_keys]
            
            cy_group_years = []
            for g in cy_group_keys:
                try:
                    yr = int(g.split("_")[0])
                except:
                    yr = 1977
                cy_group_years.append(yr)

            cy_weights = [np.exp(-abs(y - target_year) / 10) for y in cy_group_years]

            dtrain_cy = xgb.DMatrix(X_cy, label=y_cy, weight=cy_weights)
            dtrain_cy.set_group(cy_group_sizes)
            
            cy_params = {
                "objective": "rank:pairwise", "eval_metric": "ndcg", 
                "eta": 0.05, "max_depth": 4 
            }
            
            cy_model = xgb.train(cy_params, dtrain_cy, num_boost_round=200)
            st.session_state["cy_model"] = cy_model
            st.session_state["cy_features"] = cy_feats 
            st.success("Cy Young Model trained!")
            
        except FileNotFoundError:
            st.error("Missing 'cy_training_df.pkl'. Run 'train_cy_young.py' locally first.")
        except Exception as e:
            st.error(f"Cy Young Training failed: {e}")

# ------------------------
# Upload Season
# ------------------------
st.markdown("### Upload Season Data")
col1, col2 = st.columns(2)
with col1:
    player_file = st.file_uploader("1. Player Statistics (CSV)", type="csv")
with col2:
    team_file = st.file_uploader("2. Team Standings (CSV)", type="csv")

def process_input_files(player_df, team_df):
    p = player_df.copy()
    t = team_df.copy()

    # 1. Standardize Team Data
    t = t.rename(columns={"Abbr": "teamID", "SL": "lgID", "%": "winP"})
    
    # 2. Player ID & Clean Up
    if "Name" in p.columns and "ID" in p.columns:
        p["playerID"] = p["Name"].astype(str) + " (" + p["ID"].astype(str) + ")"
    else:
        p["playerID"] = p.index.astype(str)

    if "ROOK" in p.columns:
        p["ROOK"] = p["ROOK"].astype(str).str.title()

    # 3. Position Mapping
    pos_map = {1: 'P', 2: 'C', 3: '1B', 4: '2B', 5: '3B',
               6: 'SS', 7: 'LF', 8: 'CF', 9: 'RF', 0: 'OF'}
    if "POS" in p.columns:
        p["primary_pos"] = p["POS"].map(pos_map).fillna("OF")
    else:
        p["primary_pos"] = "OF"

    # 4. Handle IP
    if "IP" in p.columns:
        ip_vals = pd.to_numeric(p["IP"], errors="coerce").fillna(0.0)
        ip_int = ip_vals.astype(int)
        ip_dec = (ip_vals - ip_int) * 10 / 3
        p["IP"] = ip_int + ip_dec
    else:
        p["IP"] = 0.0

    # 5. Calculate Basic Stats
    def s(col): return pd.to_numeric(p.get(col, 0), errors="coerce").fillna(0)
    
    p["H"] = s("1B") + s("2B") + s("3B") + s("HR")
    p["X1B"] = s("1B")
    p["FO"] = s("AB") - s("H") - s("SO")
    p["BBHBP"] = s("BB") + (s("HP") if "HP" in p.columns else s("HBP"))

    p["ER"] = s("ER")
    p["SO"] = s("K") 
    p["ERA"] = np.where(p["IP"] > 0, (p["ER"] * 9) / p["IP"], 99.0)
    p["W"] = s("W")
    p["L"] = s("L")
    p["SV"] = s("SV")
    p["SHO"] = s("SHO")

    # 6. Merge Team Data
    special = ["Retired", "-", "Free Agent"]
    merged = p.merge(t, left_on="ORG", right_on="teamID", how="left", suffixes=("", "_team"))
    
    merged["winP"] = merged.apply(
        lambda x: 0.5 if x.get("ORG") in special or pd.isna(x.get("winP")) else x.get("winP"), axis=1
    )
    merged["lgID"] = merged["lgID"].fillna("FA")
    merged["teamID"] = merged["ORG"]
    
    for c in list(merged.columns):
        if c.endswith("_team"):
            base = c[:-5]
            if base in merged.columns: merged.drop(columns=[c], inplace=True)
            else: merged.rename(columns={c: base}, inplace=True)

    # 7. League Winner
    valid_mask = ~merged.ORG.isin(special) if "ORG" in merged.columns else pd.Series(False, index=merged.index)
    if valid_mask.any():
        max_wins_per_lg = merged.loc[valid_mask].groupby("lgID")["winP"].max().to_dict()
        merged["max_winP_in_lg"] = merged["lgID"].map(max_wins_per_lg).fillna(0)
        merged["LgWin"] = ((merged["winP"] == merged["max_winP_in_lg"]) & valid_mask).astype(int)
    else:
        merged["max_winP_in_lg"] = 0
        merged["LgWin"] = 0
    
    merged["WSWin"] = 0 

    # ------------------------------------------------
    # NEW: Save display position before one-hot encoding
    # ------------------------------------------------
    merged["display_pos"] = merged["primary_pos"]

    # 8. Dummy Positions
    merged["primary_pos"] = pd.Categorical(merged["primary_pos"], 
        categories=["1B","2B","3B","SS","LF","CF","RF","OF","C","P"])
    merged = pd.get_dummies(merged, columns=["primary_pos"])
    
    merged = merged.loc[:, ~merged.columns.duplicated()]

    return merged

def predict_and_display(df, model, feature_cols, title, top_n=15, position=None):
    """
    Prepare predictions and return top_n per league. If `position` is provided (string),
    filter players to that position before ranking so you get the top-N for that position.
    """
    # Prepare features
    df_pred = df.copy()
    for c in feature_cols:
        if c not in df_pred.columns:
            df_pred[c] = 0

    X = df_pred[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
    dtest = xgb.DMatrix(X)
    df_pred["pred_score"] = model.predict(dtest)

    # Apply position filter BEFORE ranking if requested
    if position and position != "All":
        if "display_pos" in df_pred.columns:
            df_pred = df_pred[df_pred["display_pos"].astype(str) == str(position)].copy()
        else:
            # no display_pos column, nothing to show
            st.warning(f"Position column not found; showing all players for {title}.")

    # Rank top_n per league
    rankings = (
        df_pred
        .sort_values(["lgID", "pred_score"], ascending=[True, False])
        .groupby("lgID", group_keys=False)
        .head(top_n)
        .assign(rank=lambda x: x.groupby("lgID").cumcount() + 1)
    )

    st.subheader(f"{title} Voting Prediction")

    # layout per league
    lg_list = list(rankings.lgID.unique())
    if not lg_list:
        st.info("No candidates to display for this selection.")
        return rankings

    cols = st.columns(len(lg_list))
    for i, lg in enumerate(lg_list):
        with cols[i]:
            st.markdown(f"**{lg}**")
            display_cols = ["rank", "playerID", "teamID", "pred_score"]
            if "display_pos" in rankings.columns:
                display_cols.insert(2, "display_pos")
            st.dataframe(
                rankings[rankings.lgID == lg][display_cols],
                width='stretch', hide_index=True
            )
    return rankings

def _plot_factor_graph(model, row_feats, feat_names, top_k=30):
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(row_feats)

        if isinstance(shap_values, list):
            sv = np.array(shap_values[0]).flatten()
        else:
            sv = np.array(shap_values).flatten()

        fnames = list(feat_names)
        if len(fnames) != sv.shape[0]:
            fnames = list(row_feats.columns)
            sv = sv[:len(fnames)]

        info = pd.DataFrame({"feature": fnames, "impact": sv})
        info["abs"] = info["impact"].abs()
        info = info.sort_values("abs", ascending=False).head(top_k).drop(columns=["abs"]).reset_index(drop=True)

        fig, ax = plt.subplots(figsize=(6, max(2, 0.25 * len(info))))
        ax.barh(info["feature"][::-1], info["impact"][::-1], color="C0")
        ax.set_xlabel("SHAP impact")
        ax.set_title(f"Top {len(info)} feature impacts")
        plt.tight_layout()

        return fig, info

    except Exception:
        return None, None

def _plot_comparison_graph(model, row1, row2, feat_names, p1_name, p2_name, top_k=15):
    """
    Compares two players and plots the features that create the biggest gap between them.
    Positive values mean the feature favors Player 1.
    Negative values mean the feature favors Player 2.
    """
    try:
        explainer = shap.TreeExplainer(model)
        
        # Calculate SHAP for both
        shap1 = explainer.shap_values(row1)[0] if isinstance(explainer.shap_values(row1), list) else explainer.shap_values(row1)
        shap2 = explainer.shap_values(row2)[0] if isinstance(explainer.shap_values(row2), list) else explainer.shap_values(row2)
        
        # Flatten
        shap1 = np.array(shap1).flatten()
        shap2 = np.array(shap2).flatten()
        
        # Calculate Difference (Advantage for Player 1)
        diff = shap1 - shap2
        
        info = pd.DataFrame({
            "feature": feat_names,
            "net_advantage": diff,
            "p1_val": row1.values.flatten(),
            "p2_val": row2.values.flatten()
        })
        
        # Sort by absolute impact of the difference
        info["abs_diff"] = info["net_advantage"].abs()
        info = info.sort_values("abs_diff", ascending=False).head(top_k)
        
        # Plot
        fig, ax = plt.subplots(figsize=(8, 0.4 * top_k))
        
        # Color bars: Green if favors P1, Red if favors P2
        colors = ["#2ca02c" if x > 0 else "#d62728" for x in info["net_advantage"][::-1]]
        
        ax.barh(info["feature"][::-1], info["net_advantage"][::-1], color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        
        ax.set_title(f"Why did the model prefer {p1_name} over {p2_name}?")
        ax.set_xlabel(f"<-- Favors {p2_name} | Favors {p1_name} -->")
        plt.tight_layout()
        
        return fig, info.drop(columns=["abs_diff"])
        
    except Exception as e:
        st.error(f"Comparison Error: {e}")
        return None, None

# ------------------------
# Execution
# ------------------------
if player_file and team_file:
    try:
        raw_p = pd.read_csv(player_file)
        raw_t = pd.read_csv(team_file)
        
        master_df = process_input_files(raw_p, raw_t)
        
        mvp_model = st.session_state.get("mvp_model")
        mvp_feats = json.load(open("feature_columns.json")) if os.path.exists("feature_columns.json") else []

        cy_model = None
        cy_feats = []
        if os.path.exists("cy_model.pkl"):
            cy_model = pickle.load(open("cy_model.pkl", "rb"))
            cy_feats = json.load(open("cy_features.json"))
        
        # --------------------
        # UI Selector
        # --------------------
        award_mode = st.radio("Select Award", ["MVP", "Rookie of the Year (ROY)", "Cy Young"], horizontal=True)
        
        ranked_df = pd.DataFrame()

        curr_model = None
        curr_feats = []

        if award_mode == "MVP":
            if mvp_model:
                curr_model = mvp_model
                curr_feats = mvp_feats
                # position dropdown for MVP
                available_pos = sorted(master_df["display_pos"].astype(str).unique()) if "display_pos" in master_df.columns else []
                pos_options = ["All"] + available_pos
                selected_pos = st.selectbox("Show top 15 for position:", pos_options, index=0, key="mvp_pos")
                ranked_df = predict_and_display(master_df, mvp_model, mvp_feats, "MVP", 15, position=selected_pos)
            else:
                st.error("MVP Model not found. Please train it above.")

        elif award_mode == "Rookie of the Year (ROY)":
            if mvp_model:
                curr_model = mvp_model
                curr_feats = mvp_feats
                roy_df = master_df[master_df["ROOK"].astype(str).str.lower() == "yes"].copy()
                if not roy_df.empty:
                    available_pos = sorted(roy_df["display_pos"].astype(str).unique()) if "display_pos" in roy_df.columns else []
                    pos_options = ["All"] + available_pos
                    selected_pos_roy = st.selectbox("Show top for position (ROY):", pos_options, index=0, key="roy_pos")
                    ranked_df = predict_and_display(roy_df, mvp_model, mvp_feats, "ROY", 5, position=selected_pos_roy)
                else:
                    st.warning("No rookies found.")
            else:
                st.error("MVP Model (used for ROY) not found.")

        elif award_mode == "Cy Young":
            if "cy_model" in st.session_state:
                cy_model = st.session_state["cy_model"]
                cy_feats = st.session_state.get("cy_features", [])
                curr_model = cy_model
                curr_feats = cy_feats

                if "primary_pos_P" in master_df.columns:
                    cy_candidates = master_df[master_df["primary_pos_P"] == 1].copy()
                    if not cy_candidates.empty:
                        # for Cy Young we keep position fixed to Pitcher but allow league split only
                        ranked_df = predict_and_display(cy_candidates, cy_model, cy_feats, "Cy Young", top_n=10, position="P")
                    else:
                        st.warning("No pitchers found.")
                else:
                    st.error("Column 'primary_pos_P' missing.")
            else:
                st.error("Cy Young Model not trained.")

        # --------------------
        # Single Player Deep Dive
        # --------------------
        if not ranked_df.empty:
            st.divider()
            
            # Create tabs for Single vs Comparison
            tab1, tab2 = st.tabs(["Single Player Deep Dive", "Head-to-Head Comparison"])
            
            with tab1:
                st.subheader("Single Player Analysis")
                selected = st.selectbox("Select Player", ranked_df.playerID.unique())
                
                if st.button("Explain Player Score"):
                    player_row = ranked_df[ranked_df.playerID == selected].iloc[0]
                    
                    row_feats = pd.DataFrame([player_row])
                    for c in curr_feats:
                        if c not in row_feats.columns: row_feats[c] = 0
                    row_feats = row_feats[curr_feats].apply(pd.to_numeric, errors="coerce").fillna(0)

                    fig, info = _plot_factor_graph(curr_model, row_feats, curr_feats, top_k=20)
                    if fig is not None:
                        st.pyplot(fig)
                        st.dataframe(info, hide_index=True)
            
            with tab2:
                # ------------------------------------------------
                # NEW: Head-to-Head Comparison
                # ------------------------------------------------
                st.subheader("Head-to-Head Comparison")
                c1, c2, c3 = st.columns([1, 1, 0.5])
                
                # Default to top 2 players if available
                p_options = list(ranked_df.playerID.unique())
                def_1 = 0
                def_2 = 1 if len(p_options) > 1 else 0
                
                p1_sel = c1.selectbox("Player A (Focus)", p_options, index=def_1)
                p2_sel = c2.selectbox("Player B (Reference)", p_options, index=def_2)
                
                if st.button("Compare Players", type="primary"):
                    # Extract rows
                    row1_full = ranked_df[ranked_df.playerID == p1_sel].iloc[0]
                    row2_full = ranked_df[ranked_df.playerID == p2_sel].iloc[0]
                    
                    st.markdown(f"**Score Gap:** {row1_full['pred_score']:.3f} vs {row2_full['pred_score']:.3f}")
                    
                    # Prepare features
                    def get_feats(r):
                        df_r = pd.DataFrame([r])
                        for c in curr_feats:
                            if c not in df_r.columns: df_r[c] = 0
                        return df_r[curr_feats].apply(pd.to_numeric, errors="coerce").fillna(0)

                    r1_feats = get_feats(row1_full)
                    r2_feats = get_feats(row2_full)
                    
                    fig, comp_info = _plot_comparison_graph(
                        curr_model, r1_feats, r2_feats, curr_feats, p1_sel, p2_sel
                    )
                    
                    if fig:
                        st.pyplot(fig)
                        st.caption("Positive (Green) bars mean this feature helped Player A beat Player B. Negative (Red) bars mean Player B was better in this area.")
                        st.dataframe(comp_info, hide_index=True)

    except Exception as e:
        st.error(f"Error: {e}")

        #streamlit run c:/Users/rowan/OneDrive/Documents/Lahman/app.py