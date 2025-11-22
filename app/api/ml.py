import torch
import torch.nn as nn
from app.models.classifier.tradeclassifier import TradeLSTMClassifier # Import the classifier model
import numpy as np
import pandas as pd
from stable_baselines3 import PPO # Import PPO for RL model
from app.models.RL.rl import NudgeExcessEnv,sessions
import os
device='cpu'
id2label = {0:"revenge",1:"fomo",2:"herd",3:"disciplined",4:"overconfident"}
NUDGES = ["dont_hesitate","calm_down","take_a_second","alert_stop"]
BATCH_SIZE = 32
EPOCHS = 1000
MAX_SEQ_LEN = 40            # clip or pad sequences to this length
EMBED_DIM = 64              # stock embedding dim
SIDE_EMB = 8                # buy/sell embedding dim
TRADE_FEAT_DIM = 9          # number of numeric features per trade
TRADE_EMB_DIM = 128
LSTM_HIDDEN = 128
NUM_CLASSES = 5
LR = 1e-4
PATIENCE = 10  

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CSV_PATH = os.path.join(
    APP_DIR,
    "models",
    "synthetic_trade_sequences_planned",
    "synthetic_trade_sequences_planned.csv"
)
try:
    classifier_model = TradeLSTMClassifier(
        stock_vocab_size=6, # Example, adjust based on your data
        trade_feat_dim=TRADE_FEAT_DIM,
        stock_emb_dim=EMBED_DIM,
        side_emb_dim=SIDE_EMB,
        trade_emb_dim=TRADE_EMB_DIM,
        lstm_hidden=LSTM_HIDDEN,
        num_classes=NUM_CLASSES
    )
    # Adjust the path as necessary
    classifier_model.load_state_dict(torch.load("app/models/classifier/best_trade_lstm.pt", map_location=torch.device('cpu')))
    classifier_model.eval()
except Exception as e:
    print(f"Error loading TradeLSTMClassifier model: {e}")
    classifier_model = None


# Load the pre-trained PPO model for nudges
try:
    # Initialize NudgeExcessEnv with an empty list of sessions. This instance is used
    # by PPO.load primarily to infer observation and action space shapes.
    # Actual session data for inference is passed directly to _make_obs.
    nudge_env = NudgeExcessEnv(sessions)
    nudge_model = PPO.load("app/models/RL/ppo_nudge_excess_model_gymnasium.zip", env=nudge_env, device="cpu")
except Exception as e:
    print(f"Error loading PPO nudge model: {e}")
    nudge_model = None


#from api.onboarding import get_onboarding_data
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
#from app.api.deps import get_current_user, get_async_db
#from app.db.models import User as DBUser, Onboarding as DBOnboarding
from app.schemas import OnboardingCreate, User, TradeData # Import TradeData



STOCK_ID_MAP = {f'STOCK_{i}': i for i in range(1000)} # Example mapping


router = APIRouter()

@router.get("/classifierdata", tags=["ml"])
async def get_classifier_data():
    df = pd.read_csv(CSV_PATH)
    df = df.drop(columns=["archetype", "label"], errors="ignore")
    return df[:100].to_dict(orient="records")


@router.get("/classifierusers", response_model=list[User], tags=["ml"])
async def classifier_users():
    return None
    
@router.post("/classifytrades", tags=["ml"])
async def classify_trades(trade_data: list[TradeData]):
    """
    trade_data = list of trades belonging to ONE sequence (ONE trading session)
    """
    if classifier_model is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Trade classifier model not loaded."
        )

    if STOCK_ID_MAP is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Stock vocab not loaded (stock2id)."
        )

    # ---------------------------
    # Build the 9-D feature vector
    # match TRADE_FEAT_DIM in training code
    # ---------------------------
    feats = []
    stock_ids = []
    side_ids = []

    for t in trade_data:
        # These features MUST MATCH the training CSV columns exactly & order
        feats.append([
            t.qty,
            t.entry_price,
            t.exit_price,
            t.pnl,
            t.position_size,
            t.leverage,
            t.time_since_prev_sec,
            t.planned_max_trades,
            t.actual_trades_in_session
        ])

        stock_ids.append(STOCK_ID_MAP.get(t.stock, 0))   # default PAD=0
        side_ids.append(1 if t.side.upper() == "BUY" else 2)

    # ---------------------------
    # Convert to tensors (batch of 1 seq)
    # ---------------------------
    L = len(trade_data)
    max_len = MAX_SEQ_LEN

    # Right-align padding to match collate_fn logic
    pad_len = max_len - L
    if pad_len < 0:
        # trim older trades, keep most recent MAX_SEQ_LEN
        feats = feats[-max_len:]
        stock_ids = stock_ids[-max_len:]
        side_ids = side_ids[-max_len:]
        L = max_len
        pad_len = 0

    # pad on the LEFT (right-align)
    feats = [[0.0]*TRADE_FEAT_DIM]*pad_len + feats
    stock_ids = [0]*pad_len + stock_ids
    side_ids = [0]*pad_len + side_ids

    # Final tensors: shape (1, L, dim)
    feats_tensor = torch.tensor([feats], dtype=torch.float32)
    stock_ids_tensor = torch.tensor([stock_ids], dtype=torch.long)
    side_ids_tensor = torch.tensor([side_ids], dtype=torch.long)
    lengths_tensor = torch.tensor([L], dtype=torch.long)

    # ---------------------------
    # Inference
    # ---------------------------
    classifier_model.eval()
    with torch.no_grad():
        logits = classifier_model(
            feats_tensor.to(device),
            stock_ids_tensor.to(device),
            side_ids_tensor.to(device),
            lengths_tensor.to(device)
        )
        pred = logits.argmax(dim=1).item()
    print(pred)
    return {
        "sequence_length": L,
        "prediction_id": pred,
        "prediction_label": id2label[pred]
    }
    

@router.post("/get_nudge", tags=["ml"])
async def get_nudge(trade_history: list[TradeData]):
    if nudge_model is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Nudge model not loaded.")
    if nudge_env is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Nudge environment not initialized.")

    # Use the env's configured max_history if present, else default to 10
    max_history = getattr(nudge_env, "max_history", 10)

    # ---- convert incoming Pydantic objects to DataFrame ----
    if not trade_history:
        trade_df = pd.DataFrame(
            columns=[
                "trade_idx","timestamp","stock","side","qty","entry_price","exit_price",
                "pnl","position_size","leverage","time_since_prev_sec",
                "planned_max_trades","actual_trades_in_session","label"
            ]
        )
    else:
        trade_df = pd.DataFrame([t.dict() for t in trade_history])

    # ---- normalize / ensure columns exist and types are correct ----
    # parse timestamps (classifier format uses ISO datetimes)
    if "timestamp" in trade_df.columns and not trade_df["timestamp"].isnull().all():
        trade_df["timestamp"] = pd.to_datetime(trade_df["timestamp"], errors="coerce")
    else:
        # if no timestamp, create synthetic increasing timestamps spaced by 1 second
        trade_df["timestamp"] = pd.to_datetime("now") + pd.to_timedelta(np.arange(len(trade_df)), unit="s")

    # ensure numeric columns exist (fill missing with 0)
    numeric_cols = ["qty","entry_price","exit_price","pnl","position_size","leverage","time_since_prev_sec"]
    for c in numeric_cols:
        if c not in trade_df.columns:
            trade_df[c] = 0.0
        trade_df[c] = pd.to_numeric(trade_df[c], errors="coerce").fillna(0.0)

    # ensure trade_idx exists, otherwise assign sequential indices starting at 1
    if "trade_idx" not in trade_df.columns or trade_df["trade_idx"].isnull().all():
        trade_df["trade_idx"] = np.arange(1, len(trade_df) + 1)
    else:
        trade_df["trade_idx"] = trade_df["trade_idx"].astype(int)

    # ensure planned_max_trades column exists and is int (take first value if present)
    if "planned_max_trades" in trade_df.columns and not trade_df["planned_max_trades"].isnull().all():
        planned_val = int(trade_df["planned_max_trades"].iloc[0])
    else:
        # fallback: if available, try `actual_trades_in_session` or length of history
        if "actual_trades_in_session" in trade_df.columns and not trade_df["actual_trades_in_session"].isnull().all():
            planned_val = int(trade_df["actual_trades_in_session"].iloc[0])
        else:
            planned_val = max(1, len(trade_df))  # default to session length

    # actual trades (prefer explicit column, else length)
    if "actual_trades_in_session" in trade_df.columns and not trade_df["actual_trades_in_session"].isnull().all():
        actual_val = int(trade_df["actual_trades_in_session"].iloc[0])
    else:
        actual_val = int(len(trade_df))

    # ensure label/archetype (optional)
    if "label" in trade_df.columns and not trade_df["label"].isnull().all():
        archetype = str(trade_df["label"].iloc[0])
    else:
        archetype = "disciplined"  # default if not provided

    # compute time_since_prev_sec if not present or unreliable
    if "time_since_prev_sec" not in trade_df.columns or trade_df["time_since_prev_sec"].isnull().any():
        trade_df = trade_df.sort_values(by=["trade_idx", "timestamp"]).reset_index(drop=True)
        trade_df["time_since_prev_sec"] = trade_df["timestamp"].diff().dt.total_seconds().fillna(0.0)

    # compute pnl if missing but entry/exit exist
    if ("pnl" not in trade_df.columns) or trade_df["pnl"].isnull().all():
        # simple pnl estimate: (exit - entry) * qty (if exists)
        trade_df["pnl"] = (trade_df["exit_price"].fillna(0.0) - trade_df["entry_price"].fillna(0.0)) * trade_df["qty"].fillna(0.0)

    # sort by trade_idx (chronological model used in env)
    trade_df = trade_df.sort_values("trade_idx").reset_index(drop=True)

    # ---- split history vs excess using planned_val ----
    # history = trades up to planned_val (<= planned), excess = trades after planned_val
    history_df = trade_df[trade_df["trade_idx"] <= planned_val].copy().reset_index(drop=True)
    if history_df.shape[0] == 0:
        # if nothing <= planned, fallback to the first MAX_HISTORY_TRADES rows as history
        history_df = trade_df.head(MAX_HISTORY_TRADES).copy().reset_index(drop=True)

    excess_df = trade_df[trade_df["trade_idx"] > planned_val].copy().reset_index(drop=True)
    baseline_excess_pnl = float(excess_df["pnl"].sum()) if excess_df.shape[0] > 0 else 0.0

    # trim history to MAX_HISTORY_TRADES (keep most recent trades)
    if history_df.shape[0] > max_history:
        history_df = history_df.tail(max_history).reset_index(drop=True)

    # ---- build session dict expected by NudgeExcessEnv ----
    session_for_nudge = {
        "sequence_id": int(trade_df["trade_idx"].min()) if len(trade_df) > 0 else 1,
        "archetype": archetype,
        "planned_max_trades": int(planned_val),
        "actual_trades": int(actual_val),
        "history_df": history_df.reset_index(drop=True),
        "excess_df": excess_df.reset_index(drop=True),
        "baseline_excess_pnl": float(baseline_excess_pnl)
    }

    # set env current session and compute observation
    nudge_env.curr = session_for_nudge
    obs = nudge_env._make_obs(session_for_nudge)

    # ensure obs is numpy array and dtype float32
    if not isinstance(obs, np.ndarray):
        obs = np.asarray(obs, dtype=np.float32)
    else:
        obs = obs.astype(np.float32)

    # predict action
    action_raw, _states = nudge_model.predict(obs, deterministic=True)
    if isinstance(action_raw, (list, tuple, np.ndarray)):
        action = int(np.asarray(action_raw).flatten()[0])
    else:
        action = int(action_raw)
    action = max(0, min(action, len(NUDGES)-1))
    nudge_action = NUDGES[action]

    return {
        "nudge": nudge_action,
        "action_id": action,
        "planned_max_trades": session_for_nudge["planned_max_trades"],
        "actual_trades": session_for_nudge["actual_trades"],
        "baseline_excess_pnl": session_for_nudge["baseline_excess_pnl"],
        "message": "Nudge generated based on RL model."
    }