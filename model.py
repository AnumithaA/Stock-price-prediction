# model.py — full file with all classes

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets,
                                  weight=self.alpha, reduction="none")
        pt    = torch.exp(-ce_loss)
        focal = (1 - pt) ** self.gamma * ce_loss
        return focal.mean()


class BiLSTMWithAttention(nn.Module):
    """
    Standalone BiLSTM — used for ablation variants
    that don't use the GCN (price-only, price+sentiment).
    Input: pre-merged feature tensor [B, T, input_dim]
    """
    def __init__(self, input_dim=44, hidden_dim=32,
                 num_layers=2, dropout=0.3, num_classes=2):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.bn        = nn.BatchNorm1d(hidden_dim * 2)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        lstm_out, _ = self.bilstm(x)
        attn_w      = torch.softmax(self.attention(lstm_out), dim=1)
        context     = (attn_w * lstm_out).sum(dim=1)
        context     = self.bn(context)
        context     = self.dropout(context)
        return self.fc(context)


class GCNBiLSTM(nn.Module):
    """
    Full model: vectorized GCN enriches cross-stock sentiment,
    then BiLSTM models the temporally weighted sequence.
    Used for the full architecture variant in the ablation study.
    """
    def __init__(self, price_dim=11, sent_dim=33,
                 gcn_hidden=64, gcn_out=33,
                 lstm_hidden=32, dropout=0.3,
                 num_stocks=2, num_classes=2):
        super().__init__()
        self.price_dim  = price_dim
        self.sent_dim   = sent_dim
        self.num_stocks = num_stocks
        self.gcn_out    = gcn_out

        # GCN layers
        self.gcn_conv1 = nn.Linear(sent_dim,   gcn_hidden)
        self.gcn_conv2 = nn.Linear(gcn_hidden, gcn_out)
        self.gcn_bn1   = nn.BatchNorm1d(gcn_hidden)
        self.gcn_bn2   = nn.BatchNorm1d(gcn_out)
        self.gcn_drop  = nn.Dropout(0.3)

        # BiLSTM
        input_dim      = price_dim + gcn_out
        self.bilstm    = nn.LSTM(input_size=input_dim,
                                  hidden_size=lstm_hidden,
                                  num_layers=2,
                                  batch_first=True,
                                  bidirectional=True,
                                  dropout=dropout)
        self.attention = nn.Linear(lstm_hidden * 2, 1)
        self.bn        = nn.BatchNorm1d(lstm_hidden * 2)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def apply_gcn_batched(self, x_sent, edge_index):
        """
        x_sent  : [B, T, num_stocks, sent_dim]
        returns : [B, T, num_stocks, gcn_out]
        """
        B, T, S, D = x_sent.shape
        x       = x_sent.reshape(B * T, S, D)
        x_neigh = torch.flip(x, dims=[1])
        x_agg   = (x + x_neigh) / 2.0

        # Layer 1
        h = self.gcn_conv1(x_agg)
        BTS, S2, H = h.shape
        h = self.gcn_bn1(h.reshape(-1, H)).reshape(BTS, S2, H)
        h = F.relu(h)
        h = self.gcn_drop(h)

        # Layer 2
        h = self.gcn_conv2(h)
        BTS, S2, G = h.shape
        h = self.gcn_bn2(h.reshape(-1, G)).reshape(BTS, S2, G)
        h = F.relu(h)

        return h.reshape(B, T, S, G)

    def predict(self, x_price, x_sent, edge_index, stock_node_idx):
        """
        x_price        : [B, T, price_dim]
        x_sent         : [B, T, num_stocks, sent_dim]
        edge_index     : [2, num_edges]
        stock_node_idx : int  0=AAPL, 1=TSLA
        """
        gcn_out    = self.apply_gcn_batched(x_sent, edge_index)
        stock_sent = gcn_out[:, :, stock_node_idx, :]
        combined   = torch.cat([x_price, stock_sent], dim=-1)

        lstm_out, _ = self.bilstm(combined)
        attn_w      = torch.softmax(self.attention(lstm_out), dim=1)
        context     = (attn_w * lstm_out).sum(dim=1)
        context     = self.bn(context)
        context     = self.dropout(context)
        return self.fc(context)