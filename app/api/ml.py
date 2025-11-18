from api.onboarding import get_onboarding_data
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.api.deps import get_current_user, get_async_db
from app.db.models import User as DBUser, Onboarding as DBOnboarding
from app.schemas import OnboardingCreate, User
import torch
import torch.nn as nn


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
async def classify_trades():


class TradeLSTMClassifier(nn.Module):
    def __init__(self, stock_vocab_size, trade_feat_dim=TRADE_FEAT_DIM,
                 stock_emb_dim=EMBED_DIM, side_emb_dim=SIDE_EMB,
                 trade_emb_dim=TRADE_EMB_DIM, lstm_hidden=LSTM_HIDDEN,
                 num_classes=NUM_CLASSES):
        super().__init__()
        self.stock_emb = nn.Embedding(stock_vocab_size+1, stock_emb_dim, padding_idx=0)
        self.side_emb = nn.Embedding(3, side_emb_dim, padding_idx=0)  # PAD, BUY, SELL
        self.trade_fc = nn.Sequential(
            nn.Linear(trade_feat_dim + stock_emb_dim + side_emb_dim, trade_emb_dim),
            nn.ReLU(),
            nn.LayerNorm(trade_emb_dim)
        )
        self.lstm = nn.LSTM(input_size=trade_emb_dim, hidden_size=lstm_hidden // 2,
                            num_layers=1, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, feats, stock_ids, side_ids, lengths):
        # feats: (B,L,F), stock_ids/side_ids: (B,L)
        s_emb = self.stock_emb(stock_ids)    # (B,L,Es)
        side_e = self.side_emb(side_ids)     # (B,L,Es2)
        x = torch.cat([feats, s_emb, side_e], dim=-1)  # (B,L,Feat+Es+Es2)
        x = self.trade_fc(x)                 # (B,L,trade_emb)
        packed_out, _ = self.lstm(x)        # (B,L,hidden) because batch_first=True
        device = feats.device
        L = x.size(1)
        # create mask to ignore padded positions (right-aligned)
        mask = (torch.arange(L, device=device).unsqueeze(0) >= (L - lengths.unsqueeze(1))).float()  # (B,L)
        mask = mask.unsqueeze(-1)  # (B,L,1)
        # masked mean pooling
        summed = (packed_out * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)             
        pooled = summed / denom
        logits = self.classifier(pooled)
        return logits
