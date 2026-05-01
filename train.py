from model import GCNBiLSTM, FocalLoss
from graph_builder import EDGE_INDEX
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, matthews_corrcoef, confusion_matrix
import numpy as np, torch, torch.nn as nn, joblib, os
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
edge_index = EDGE_INDEX.to(device)

TICKER_NODE = {"AAPL": 0, "TSLA": 1}
HPARAMS = {
    "AAPL": {"lr": 5e-4,  "lstm_hidden": 32, "dropout": 0.3,
             "epochs": 60, "gamma": 0.5,  "weight_decay": 1e-4,
             "patience": 12},
    "TSLA": {"lr": 3e-4,  "lstm_hidden": 32, "dropout": 0.3,
             "epochs": 60, "gamma": 1.0,  "weight_decay": 1e-4,
             "patience": 12},
}
PATIENCE = 20

def to_loader(Xp, Xs, y, batch=32, shuffle=False):
    ds = TensorDataset(torch.FloatTensor(Xp),
                       torch.FloatTensor(Xs),
                       torch.LongTensor(y))
    return DataLoader(ds, batch_size=batch, shuffle=shuffle)

results = {}

for ticker in ["AAPL", "TSLA"]:
    print(f"\n{'='*52}\n  {ticker}\n{'='*52}")
    hp         = HPARAMS[ticker]
    node_idx   = TICKER_NODE[ticker]

    X_price = np.load(f"data/{ticker}_Xprice.npy")  # [N, 20, 11]
    X_sent  = np.load(f"data/{ticker}_Xsent.npy")   # [N, 20, 2, 33]
    y       = np.load(f"data/{ticker}_y.npy")

    # Normalize price features only
    N, T, Fp = X_price.shape
    scaler   = StandardScaler()
    X_price_scaled = scaler.fit_transform(
                         X_price.reshape(-1, Fp)).reshape(N, T, Fp)
    joblib.dump(scaler, f"data/scaler_{ticker}.pkl")

    # Chronological 80/20 split
    split      = int(0.80 * N)
    val_split  = int(0.85 * split)

    def make_splits(Xp, Xs, y, split, val_split):
        return (Xp[:val_split],      Xs[:val_split],      y[:val_split],
                Xp[val_split:split], Xs[val_split:split], y[val_split:split],
                Xp[split:],          Xs[split:],          y[split:])

    (Xp_tr, Xs_tr, y_tr,
     Xp_val, Xs_val, y_val,
     Xp_te, Xs_te, y_te) = make_splits(
         X_price_scaled, X_sent, y, split, val_split)

    print(f"Inner train {Xp_tr.shape[0]} | "
          f"Val {Xp_val.shape[0]} | Test {Xp_te.shape[0]}")
    print(f"Balance — train {y_tr.mean():.2f} | "
          f"val {y_val.mean():.2f} | test {y_te.mean():.2f}")

    weights = compute_class_weight("balanced",
                                    classes=np.unique(y_tr), y=y_tr)
    alpha   = torch.FloatTensor(weights).to(device)
    loss_fn = FocalLoss(alpha=alpha, gamma=hp["gamma"])
    print(f"Class weights: Down={weights[0]:.3f}, Up={weights[1]:.3f}")

    train_loader = to_loader(Xp_tr,  Xs_tr,  y_tr,  shuffle=True)
    val_loader   = to_loader(Xp_val, Xs_val, y_val)
    test_loader  = to_loader(Xp_te,  Xs_te,  y_te)

    model = GCNBiLSTM(price_dim=11, sent_dim=33,
                       gcn_hidden=64, gcn_out=33,
                       lstm_hidden=hp["lstm_hidden"],
                       dropout=hp["dropout"]).to(device)

    opt = torch.optim.Adam(model.parameters(),
                            lr=hp["lr"],
                            weight_decay=hp["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    opt, mode="max", factor=0.5, patience=6)

    best_val_mcc, best_val_acc = -1.0, 0.0
    epochs_no_improve = 0

    for epoch in range(hp["epochs"]):
        # LR warmup
        if epoch < 5:
            for pg in opt.param_groups:
                pg["lr"] = hp["lr"] * ((epoch + 1) / 5)
        elif epoch == 5:
            for pg in opt.param_groups:
                pg["lr"] = hp["lr"]

        # Train
        model.train()
        total_loss = 0
        for xp, xs, yb in train_loader:
            xp, xs, yb = xp.to(device), xs.to(device), yb.to(device)
            opt.zero_grad()
            logits = model.predict(xp, xs, edge_index, node_idx)
            loss   = loss_fn(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()

        # Validate
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for xp, xs, yb in val_loader:
                probs = torch.softmax(
                    model.predict(xp.to(device), xs.to(device),
                                  edge_index, node_idx), dim=1)
                preds = (probs[:, 1] > 0.50).long().cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(yb.numpy())

        val_acc    = np.mean(np.array(val_preds) == np.array(val_labels))
        val_mcc    = matthews_corrcoef(val_labels, val_preds)
        avg_loss   = total_loss / len(train_loader)
        current_lr = opt.param_groups[0]["lr"]
        scheduler.step(val_acc)

        both_classes = len(np.unique(val_preds)) == 2
        never_saved  = best_val_mcc == -1.0
        improved     = both_classes and (val_mcc > best_val_mcc or never_saved)

        if improved:
            best_val_mcc, best_val_acc = val_mcc, val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(),
                       f"data/best_model_{ticker}.pt")
        else:
            epochs_no_improve += 1

        if (epoch + 1) % 10 == 0 or improved:
            flag      = " ← best" if improved else ""
            collapsed = "" if both_classes else " [COLLAPSED]"
            print(f"  Ep {epoch+1:03d} | loss {avg_loss:.4f} | "
                  f"val_acc {val_acc:.3f} | val_MCC {val_mcc:.4f} | "
                  f"lr {current_lr:.1e}{flag}{collapsed}")

        if epochs_no_improve >= PATIENCE:
            print(f"  Early stop ep {epoch+1} — "
                  f"best val MCC {best_val_mcc:.4f}")
            break

    # Final evaluation
    print(f"\n── Final Evaluation: {ticker} ──")
    model.load_state_dict(torch.load(f"data/best_model_{ticker}.pt"))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xp, xs, yb in test_loader:
            probs = torch.softmax(
                model.predict(xp.to(device), xs.to(device),
                              edge_index, node_idx), dim=1)
            preds = (probs[:, 1] > 0.50).long().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(yb.numpy())

    print(classification_report(all_labels, all_preds,
                                 target_names=["Down","Up"],
                                 zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(all_labels, all_preds))
    test_mcc = matthews_corrcoef(all_labels, all_preds)
    print(f"Test MCC: {test_mcc:.4f}")
    results[ticker] = {"val_mcc": best_val_mcc,
                        "test_mcc": test_mcc,
                        "val_acc": best_val_acc}

print(f"\n── Summary ──")
for t, r in results.items():
    print(f"{t}: val_acc={r['val_acc']:.3f} | "
          f"val_MCC={r['val_mcc']:.4f} | "
          f"test_MCC={r['test_mcc']:.4f}")