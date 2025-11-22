import torch
import torch.nn as nn

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

#To execute the model the data should be in the format
'''
sequence_id,trade_idx,timestamp,stock,side,qty,entry_price,exit_price,pnl,position_size,leverage,time_since_prev_sec,planned_max_trades,actual_trades_in_session,label
1,0,2024-01-01 09:30:00,AAPL,BUY,10,150.0,151.2,12,1500,2,0,10,13,revenge
1,1,2024-01-01 09:31:10,AAPL,SELL,10,151.2,152.0,8,1500,2,70,10,13,revenge
1,2,2024-01-01 09:32:40,TSLA,BUY,5,700.0,690.0,-50,3500,3,90,10,13,revenge
2,0,2024-01-02 10:00:00,MSFT,BUY,2,310.0,313.0,6,620,1,0,5,5,disciplined
2,1,2024-01-02 10:05:00,MSFT,SELL,2,313.0,314.0,2,620,1,300,5,5,disciplined

model = TradeLSTMClassifier()
weights = torch.load("model_weights.pth", map_location=torch.device('cpu'))
model.load_state_dict(weights)
model.eval()
with torch.no_grad():
'''