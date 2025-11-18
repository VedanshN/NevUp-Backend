import torch
import torch.nn as nn
from app.models.classifier.tradeclassifier import TradeLSTMClassifier # Import the classifier model
from stable_baselines3 import PPO # Import PPO for RL model
import numpy as np
import pandas as pd
from app.models.RL.rl import NudgeExcessEnv, ARCHES, NUDGES # Import RL environment and constants


# Constants for TradeLSTMClassifier
TRADE_FEAT_DIM = 10  # Placeholder, adjust as needed
EMBED_DIM = 64      # Placeholder, adjust as needed
SIDE_EMB = 16       # Placeholder, adjust as needed
TRADE_EMB_DIM = 128 # Placeholder, adjust as needed
LSTM_HIDDEN = 256   # Placeholder, adjust as needed
NUM_CLASSES = 5     # Placeholder, adjust as needed (e.g., number of trade archetypes)


# Load the pre-trained TradeLSTMClassifier model
# Make sure the path to your model weights is correct
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
    # We need an instance of the environment to load the model
    # For now, we'll create a dummy one. In a real scenario, you'd likely
    # pass session data to the env at inference time.
    dummy_sessions = [] # You'll need to provide actual session data here if not pre-loaded
    nudge_env = NudgeExcessEnv(dummy_sessions)
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
from app.schemas import OnboardingCreate, User


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
    # Define expected input for trade classification
    # This will depend on how your TradeLSTMClassifier expects its input
    # For example, you might expect a list of trade dictionaries
    # For now, let's assume a placeholder input `trade_data: dict`
):
    if classifier_model is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Trade classifier model not loaded.")
    
    # Preprocess trade_data to match model input format (feats, stock_ids, side_ids, lengths)
    # This is a placeholder and needs to be implemented based on your data format
    # For demonstration purposes, let's assume `trade_data` is already processed into tensors
    
    # Example placeholder for model input
    # feats = torch.randn(1, 10, TRADE_FEAT_DIM) # Batch_size, Sequence_length, Features
    # stock_ids = torch.randint(1, 1000, (1, 10))
    # side_ids = torch.randint(1, 3, (1, 10))
    # lengths = torch.tensor([10])

    # with torch.no_grad():
    #     logits = classifier_model(feats, stock_ids, side_ids, lengths)
    #     predictions = torch.argmax(logits, dim=1)
    
    return {"message": "Trade classification endpoint, model loaded.", "predictions": "TODO: Implement actual classification logic."}


@router.post("/get_nudge", tags=["ml"])
async def get_nudge(
    # Define expected input for nudge generation
    # This will likely be historical trade data for the user
    trade_history: list[dict] # Example input
):
    if nudge_model is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Nudge model not loaded.")

    # Convert trade_history into a format usable by NudgeExcessEnv to create an observation
    # This will involve creating a 'session' dictionary similar to how it's done in rl.py
    # For demonstration, we'll use a placeholder for observation
    
    # You'll need to implement the logic to convert `trade_history` into a `session` object
    # that `NudgeExcessEnv` can use to generate an observation.
    # This would involve creating a pandas DataFrame from trade_history and then calling
    # a function similar to `build_sessions_with_excess` from `rl.py` to get a session.
    
    # Placeholder for observation
    # For now, let's assume we can generate a dummy observation
    # In a real scenario, you'd create a NudgeExcessEnv instance with the user's data
    # and then call its _make_obs method.
    
    # Dummy observation for now
    obs = np.random.rand(nudge_env.obs_dim).astype(np.float32)

    action, _states = nudge_model.predict(obs, deterministic=True)
    nudge_action = NUDGES[int(action)]

    return {"nudge": nudge_action, "message": "Nudge generated based on RL model."}
    

