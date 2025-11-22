import torch
import torch.nn as nn
from app.models.classifier.tradeclassifier import TradeLSTMClassifier # Import the classifier model
import numpy as np
import pandas as pd
from stable_baselines3 import PPO # Import PPO for RL model
from app.models.RL.rl import NudgeExcessEnv, ARCHES, NUDGES # Import RL environment and constants


# Constants for TradeLSTMClassifier
TRADE_FEAT_DIM = 4  # Adjusted to match the number of features used in preprocessing
EMBED_DIM = 64      # Placeholder, adjust as needed
SIDE_EMB = 16       # Placeholder, adjust as needed
TRADE_EMB_DIM = 128 # Placeholder, adjust as needed
LSTM_HIDDEN = 256   # Placeholder, adjust as needed
NUM_CLASSES = 5     # Placeholder, adjust as needed (e.g., number of trade archetypes)


try:
    classifier_model = TradeLSTMClassifier(
        stock_vocab_size=1000, # Example, adjust based on your data
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
    nudge_env = NudgeExcessEnv([])
    nudge_model = PPO.load("app/models/RL/ppo_nudge_excess_model_gymnasium.zip", env=nudge_env, device="cpu")
except Exception as e:
    print(f"Error loading PPO nudge model: {e}")
    nudge_model = None


from api.onboarding import get_onboarding_data
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.api.deps import get_current_user, get_async_db
from app.db.models import User as DBUser, Onboarding as DBOnboarding
from app.schemas import OnboardingCreate, User, TradeData # Import TradeData



STOCK_ID_MAP = {f'STOCK_{i}': i for i in range(1000)} # Example mapping


router = APIRouter()

@router.get("/classifierdata", response_model=OnboardingCreate, tags=["ml"])
async def get_classifier_data(
    current_user: DBUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    onboarding_data = await get_onboarding_data(current_user.id, db)
    if not onboarding_data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Onboarding data not found for the user.")
    return onboarding_data   


@router.get("/classifierusers", response_model=list[User], tags=["ml"])
async def classifier_users():
    return None
    

@router.post("/classifytrades", tags=["ml"])
async def classify_trades(
    trade_data: list[TradeData]
):
    if classifier_model is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Trade classifier model not loaded.")
    
    # Preprocess trade_data to match model input format (feats, stock_ids, side_ids, lengths)
    feats = []
    stock_ids = []
    side_ids = []
    # Placeholder for 'lengths' - assuming all sequences are of max length for now or padding will be handled
    # For a real scenario, you'd calculate actual lengths and pad.
    
    for trade in trade_data:
        # Example: Using price, amount, cost, and a dummy feature as TRADE_FEAT_DIM = 4
        # You'll need to adapt this based on what your model truly expects as features
        feats.append([trade.price, trade.amount, trade.cost, trade.price * trade.amount]) # Example features
        stock_ids.append(STOCK_ID_MAP.get(trade.symbol, 0))  # Map symbol to ID, default to 0 if not found
        side_ids.append(1 if trade.side.lower() == 'buy' else 2) # 1 for buy, 2 for sell (0 for padding)
    
    # Convert to tensors. Assuming a batch size of 1 for now.
    feats_tensor = torch.tensor([feats], dtype=torch.float32)
    stock_ids_tensor = torch.tensor([stock_ids], dtype=torch.long)
    side_ids_tensor = torch.tensor([side_ids], dtype=torch.long)
    lengths_tensor = torch.tensor([len(trade_data)], dtype=torch.long)

    with torch.no_grad():
        logits = classifier_model(feats_tensor, stock_ids_tensor, side_ids_tensor, lengths_tensor)
        predictions = torch.argmax(logits, dim=1)
    
    return {"message": "Trade classification successful.", "predictions": predictions.tolist()}
    

@router.post("/get_nudge", tags=["ml"])
async def get_nudge(
    trade_history: list[TradeData] # Use the defined TradeData schema
):
    if nudge_model is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Nudge model not loaded.")

    # The following logic converts `trade_history` into a session object suitable
    # for NudgeExcessEnv and generates an observation for the model.
    
    # Convert TradeData list to a pandas DataFrame
    trade_df = pd.DataFrame([trade.dict() for trade in trade_history])
    
    if not trade_df.empty:
        # Ensure timestamp is in datetime format and sort
        trade_df['timestamp'] = pd.to_datetime(trade_df['timestamp'], unit='ms')
        trade_df = trade_df.sort_values(by='timestamp').reset_index(drop=True)

        # Calculate 'pnl' and 'time_since_prev_sec'
        # Assuming 'price' is entry_price and 'cost' can be used to infer exit_price or pnl
        # This is a simplification and might need adjustment based on actual trade data structure
        trade_df['pnl'] = trade_df.apply(lambda row: row['cost'] if row['side'] == 'sell' else -row['cost'], axis=1) # Placeholder: adjust as per actual PnL calculation
        trade_df['entry_price'] = trade_df['price'] # Using price as entry_price
        trade_df['qty'] = trade_df['amount'] # Using amount as quantity

        trade_df['time_since_prev_sec'] = trade_df['timestamp'].diff().dt.total_seconds().fillna(0)
    else:
        # Create an empty DataFrame with expected columns if no trade history
        trade_df = pd.DataFrame(columns=['timestamp', 'pnl', 'entry_price', 'qty', 'time_since_prev_sec', 'side'])
    
    # Create a dummy session object for NudgeExcessEnv
    # For inference, we can use the entire trade_history as 'history_df'
    # and set planned/actual trades based on its length.
    session_for_nudge = {
        "sequence_id": 1, # Dummy ID
        "archetype": "disciplined", # Placeholder, could be predicted or default
        "planned_max_trades": len(trade_history) + 5, # Assume some planned trades beyond actual
        "actual_trades": len(trade_history),
        "history_df": trade_df,
        "excess_df": pd.DataFrame(), # Not used for observation, but expected by env
        "baseline_excess_pnl": 0.0 # Not used for observation, but expected by env
    }

    # Temporarily set the environment's current session for observation generation
    # In a more robust system, you might pass the session directly to a helper function
    nudge_env.curr = session_for_nudge
    obs = nudge_env._make_obs(session_for_nudge)

    action, _states = nudge_model.predict(obs, deterministic=True)
    nudge_action = NUDGES[int(action)]

    return {"nudge": nudge_action, "message": "Nudge generated based on RL model."}