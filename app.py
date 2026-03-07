import gradio as gr
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import seaborn as sns

rf = pickle.load(open("model.pkl", "rb"))
opponents = pickle.load(open("opponents.pkl", "rb"))
venues = pickle.load(open("venues.pkl", "rb"))
matches = pd.read_csv("matches.csv", index_col=0)

matches["date"] = pd.to_datetime(matches["date"])
matches["opp_code"] = matches["opponent"].astype("category").cat.codes
matches["venue_code"] = matches["venue"].astype("category").cat.codes
matches["hour"] = matches["time"].str.replace(":.+", "", regex=True).astype("int")
matches["day_code"] = matches["date"].dt.dayofweek
matches["target"] = (matches["result"] == "W").astype("int")

test = matches[matches["date"] > '2022-01-01']
predictors = ["venue_code", "opp_code", "hour", "day_code"]
test_preds = rf.predict(test[predictors])
teams = sorted(matches["team"].unique().tolist())

def predict(team, opponent, venue, hour, day):
    opp_code = opponents.index(opponent)
    venue_code = venues.index(venue)
    day_code = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"].index(day)

    input_data = pd.DataFrame([[venue_code, opp_code, hour, day_code]],
                              columns=["venue_code", "opp_code", "hour", "day_code"])

    pred = rf.predict(input_data)[0]
    proba = rf.predict_proba(input_data)[0]

    # Win probability bar chart
    fig1, ax1 = plt.subplots(figsize=(5, 2))
    fig1.patch.set_facecolor("#0d1117")
    ax1.set_facecolor("#161b22")
    ax1.barh(["No Win", "Win"], [proba[0], proba[1]], color=["#e74c3c", "#00ff85"])
    ax1.set_xlim(0, 1)
    ax1.set_xlabel("Probability", color="white")
    ax1.set_title(f"{team} vs {opponent}", color="#00ff85", fontweight="bold")
    ax1.tick_params(colors="white")
    for spine in ax1.spines.values():
        spine.set_edgecolor("#30363d")
    for i, v in enumerate([proba[0], proba[1]]):
        ax1.text(v + 0.01, i, f"{v:.1%}", va="center", color="white")
    plt.tight_layout()

    # Confusion matrix
    cm = confusion_matrix(test["target"], test_preds)
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    fig2.patch.set_facecolor("#0d1117")
    ax2.set_facecolor("#161b22")
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=["No Win", "Win"],
                yticklabels=["No Win", "Win"], ax=ax2,
                linecolor="#30363d", linewidths=0.5)
    ax2.set_xlabel("Predicted", color="white")
    ax2.set_ylabel("Actual", color="white")
    ax2.set_title("Model Confusion Matrix", color="#00ff85", fontweight="bold")
    ax2.tick_params(colors="white")
    plt.tight_layout()

    # Recent form
    team_matches = matches[matches["team"] == team].sort_values("date", ascending=False).head(5)
    form = " | ".join([
        f"{'✅' if r == 'W' else '❌' if r == 'L' else '🟡'} vs {o}"
        for r, o in zip(team_matches["result"], team_matches["opponent"])
    ])

    # Head to head
    h2h = matches[(matches["team"] == team) & (matches["opponent"] == opponent)]
    if len(h2h) == 0:
        h2h_text = "No head to head data available"
    else:
        wins = len(h2h[h2h["result"] == "W"])
        losses = len(h2h[h2h["result"] == "L"])
        draws = len(h2h[h2h["result"] == "D"])
        h2h_text = f"{team} vs {opponent} — W: {wins} | D: {draws} | L: {losses}"

    result = "✅ Win" if pred == 1 else "❌ No Win"
    precision = precision_score(test["target"], test_preds)
    recall = recall_score(test["target"], test_preds)
    stats = f"Prediction: {result} | Confidence: {proba[pred]:.1%}\nModel Precision: {precision:.1%} | Recall: {recall:.1%}"

    return fig1, fig2, f"Last 5: {form}", h2h_text, stats

custom_css = """
    .gradio-container {
        font-family: 'Inter', sans-serif;
        background-color: #0d1117;
        color: #ffffff;
    }
    .gr-button {
        background: linear-gradient(135deg, #38003c, #00ff85);
        border: none;
        color: white;
        font-weight: bold;
        border-radius: 8px;
    }
    .gr-button:hover {
        opacity: 0.9;
        transform: scale(1.02);
    }
    h1 {
        color: #00ff85 !important;
        font-size: 2em !important;
        font-weight: 800 !important;
        letter-spacing: 1px;
    }
    .gr-input, .gr-dropdown {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        color: white !important;
        border-radius: 8px;
    }
    label {
        color: #00ff85 !important;
        font-weight: 600 !important;
    }
    .gr-panel {
        background-color: #161b22 !important;
        border-radius: 12px;
        border: 1px solid #30363d;
    }
"""

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Dropdown(choices=teams, label="Your Team"),
        gr.Dropdown(choices=opponents, label="Opponent"),
        gr.Dropdown(choices=venues, label="Venue"),
        gr.Slider(minimum=8, maximum=21, step=1, label="Match Hour (24hr)"),
        gr.Dropdown(choices=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"], label="Day")
    ],
    outputs=[
        gr.Plot(label="Win Probability"),
        gr.Plot(label="Confusion Matrix"),
        gr.Text(label="Recent Form"),
        gr.Text(label="Head to Head"),
        gr.Text(label="Stats")
    ],
    title="⚽ Premier League Match Predictor",
    description="Select your team, opponent, and match details to predict the outcome using a Random Forest model trained on historical Premier League data.",
    css=custom_css,
    theme=gr.themes.Base(),
    flagging_mode = "never"
)

demo.launch()