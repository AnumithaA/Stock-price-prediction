# graph_builder.py
import torch
import numpy as np

# AAPL = node 0, TSLA = node 1
# Bidirectional edge: AAPL <-> TSLA
EDGE_INDEX = torch.tensor([[0, 1],
                            [1, 0]], dtype=torch.long)

# If you expand to more stocks later, add edges here:
# MSFT=2, GOOGL=3
# e.g. [[0,1,0,2],[1,0,2,0]] for AAPL<->TSLA and AAPL<->MSFT

TICKER_TO_NODE = {"AAPL": 0, "TSLA": 1}


def build_graph_features(sent_vecs: dict, gcn_model, device):
    """
    sent_vecs: dict of {ticker: np.array of shape [T, sent_dim]}
                one entry per stock, T = sequence length (20 days)
    gcn_model: trained SentimentGCN
    device:    torch device

    Returns: dict of {ticker: np.array of shape [T, gcn_out_dim]}
             GCN-enriched sentiment sequence per stock
    """
    tickers   = list(sent_vecs.keys())
    sent_dim  = sent_vecs[tickers[0]].shape[1]  # e.g. 33
    T         = sent_vecs[tickers[0]].shape[0]  # sequence length = 20
    num_nodes = len(tickers)

    edge_index = EDGE_INDEX.to(device)
    enriched   = {t: [] for t in tickers}

    gcn_model.eval()
    with torch.no_grad():
        for t_step in range(T):
            # Build node feature matrix for this timestep
            # Shape: [num_stocks, sent_dim]
            node_feats = torch.zeros(num_nodes, sent_dim, device=device)
            for ticker in tickers:
                node_id = TICKER_TO_NODE[ticker]
                node_feats[node_id] = torch.tensor(
                    sent_vecs[ticker][t_step], dtype=torch.float32,
                    device=device)

            # GCN forward pass
            out = gcn_model(node_feats, edge_index)  # [num_stocks, out_dim]

            for ticker in tickers:
                node_id = TICKER_TO_NODE[ticker]
                enriched[ticker].append(
                    out[node_id].cpu().numpy())

    # Stack timesteps back
    return {t: np.stack(enriched[t]) for t in tickers}
    # Each value: [T, gcn_out_dim]