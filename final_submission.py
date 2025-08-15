# -*- coding: utf-8 -*-
"""
NIFTY100 (CNX100) – ARIMA, LSTM & RL (DQN) strategies
Train on ~past-year; evaluate on most recent ~6 weeks (30 trading days).
Saves plots to ./outputs and prints a performance table.

Requirements:
pip install yfinance pandas numpy scipy statsmodels pmdarima matplotlib scikit-learn tensorflow torch
"""
import matplotlib
matplotlib.use("Agg")

import os
import math
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from datetime import datetime, timedelta, timezone

# Stats & ML
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import pmdarima as pm

# Deep Learning (LSTM)
import tensorflow as tf
from tensorflow.keras import layers, models

# RL (PyTorch DQN)
import torch
import torch.nn as nn
import torch.optim as optim
torch.set_num_threads(1)

# ---------------------- CONFIG ----------------------
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
torch.manual_seed(SEED)

INDEX_SYMBOL = "^CNX100"      # NIFTY 100 on Yahoo Finance
EVAL_DAYS    = 30             # ~6 weeks trading days
TRAIN_MIN_DAYS = 180          # ensure reasonable training history
TXN_COST_BPS = 10             # 10 bps per trade (round-trip approximated per position change)

USE_LSTM = True               # set False if TensorFlow is not available
USE_RL   = True               # set False to skip RL
OUTDIR   = "outputs"
os.makedirs(OUTDIR, exist_ok=True)

# ---------------------- UTILITIES ----------------------
def download_data(symbol=INDEX_SYMBOL, lookback_days=400):
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)
    df = yf.download(symbol, start=start.date(), end=end.date(), interval="1d", auto_adjust=True)
    df = df.dropna()
    return df

def add_indicators(df):
    out = df.copy()
    out["ret"] = out["Close"].pct_change()
    out["logret"] = np.log1p(out["ret"])
    out["sma_5"] = out["Close"].rolling(5).mean()
    out["sma_20"] = out["Close"].rolling(20).mean()
    out["sma_ratio"] = out["sma_5"] / out["sma_20"] - 1
    # RSI(14)
    delta = out["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-12)
    out["rsi14"] = 100 - (100 / (1 + rs))
    # Volatility
    out["vol_10"] = out["ret"].rolling(10).std()
    out["vol_20"] = out["ret"].rolling(20).std()
    out.dropna(inplace=True)
    return out

def split_train_test(df, eval_days=EVAL_DAYS):
    if len(df) < eval_days + TRAIN_MIN_DAYS:
        raise ValueError(f"Not enough data ({len(df)} rows). Need at least {eval_days + TRAIN_MIN_DAYS}.")
    train = df.iloc[:-eval_days].copy()
    test  = df.iloc[-eval_days:].copy()
    return train, test

def apply_txn_costs(positions):
    # Approximate cost = TXN_COST_BPS * abs(delta position)
    # positions is 0/1 daily; cost applied when position changes
    changes = positions.diff().abs().fillna(0.0)
    # Convert bps to decimal
    cost = (TXN_COST_BPS / 1e4) * changes
    return cost

def evaluate_strategy(price_df, daily_signal):
    """
    price_df must contain 'ret' (simple return).
    daily_signal: alignment with price_df index, values in {0,1} (flat/long).
    Returns performance dict and equity curve (pd.Series).
    """
    sig = daily_signal.reindex(price_df.index).fillna(0.0).clip(0, 1)
    gross = sig * price_df["ret"]
    costs = apply_txn_costs(sig)
    net = gross - costs
    equity = (1 + net).cumprod()
    # Metrics
    ann_factor = 252.0
    mu = net.mean() * ann_factor
    sigma = net.std(ddof=0) * np.sqrt(ann_factor)
    sharpe = mu / (sigma + 1e-12)
    downside = net[net < 0].std(ddof=0) * np.sqrt(ann_factor)
    sortino = mu / (downside + 1e-12)
    # Max Drawdown
    roll_max = equity.cummax()
    drawdown = equity / roll_max - 1.0
    mdd = drawdown.min()
    calmar = mu / (abs(mdd) + 1e-12)
    hitrate = (net > 0).mean()
    turnover = sig.diff().abs().sum() / len(sig)
    return {
        "AnnReturn": mu,
        "AnnVol": sigma,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "MaxDD": mdd,
        "Calmar": calmar,
        "HitRate": hitrate,
        "Turnover": turnover,
        "Cumulative": equity.iloc[-1] - 1.0
    }, equity

def print_leaderboard(results):
    df = pd.DataFrame(results).T
    cols = ["Cumulative", "AnnReturn", "AnnVol", "Sharpe", "Sortino", "MaxDD", "Calmar", "HitRate", "Turnover", "RMSE"]
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    df_sorted = df.sort_values("Cumulative", ascending=False)
    pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
    print("\n=== Strategy Leaderboard (sorted by Cumulative return) ===\n")
    print(df_sorted[cols])
    return df_sorted

def plot_equity_curves(curves_dict, title, outfile):
    plt.figure(figsize=(10,6))
    for name, eq in curves_dict.items():
        plt.plot(eq.index, eq.values, label=name)
    plt.legend()
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Equity (normalized)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outfile, dpi=130)
    plt.close()

# ---------------------- DATA PREP ----------------------
raw = download_data(INDEX_SYMBOL, lookback_days=420)
data = add_indicators(raw)
train, test = split_train_test(data, eval_days=EVAL_DAYS)

# ---------------------- BENCHMARKS ----------------------
bench_results = {}
bench_curves  = {}

# Buy & Hold (test only)
bh_sig = pd.Series(1.0, index=test.index)
bench_results["BuyHold"], bh_eq = evaluate_strategy(test, bh_sig)
bench_curves["BuyHold"] = bh_eq

# SMA(5 > 20) crossover
sma_sig = (test["sma_5"] > test["sma_20"]).astype(float)
bench_results["SMA_5over20"], sma_eq = evaluate_strategy(test, sma_sig)
bench_curves["SMA_5over20"] = sma_eq

'''# ---------------------- ARIMA MODEL (price-based, robust) ----------------------
import pmdarima as pm
import numpy as np
import pandas as pd
import math
from sklearn.metrics import mean_squared_error

arima_name = "ARIMA"

try:
    # ------------------ Fit ARIMA on training prices ------------------
    train_prices = train["Close"].dropna().astype(float).values  # 1D array
    model = pm.auto_arima(train_prices, seasonal=False, stepwise=True,
                          suppress_warnings=True, error_action="ignore",
                          max_p=5, max_q=5, stationary=False)

    # ------------------ Forecast all test prices at once ------------------
    n_test = len(test)
    pred_prices = model.predict(n_periods=n_test)
    pred_prices = np.array(pred_prices).flatten()  # ensure 1D

    # ------------------ Compute predicted returns without concatenate ------------------
    # Returns = (current_pred_price / previous_price) - 1
    # Use numpy shift: previous price = last train price + pred_prices[:-1]
    prev_prices = np.empty_like(pred_prices)
    prev_prices[0] = train_prices[-1]          # first previous price = last train price
    prev_prices[1:] = pred_prices[:-1]         # rest = previous predicted price

    pred_returns = (pred_prices / prev_prices) - 1
    pred_returns = pd.Series(pred_returns, index=test.index, name="arima_ret_pred")

    print(f"ARIMA predicted return range: {pred_returns.min():.6f} to {pred_returns.max():.6f}")

    # ------------------ Generate long-only signals ------------------
    arima_sig = (pred_returns > 0).astype(float)
    print("ARIMA signal counts:\n", arima_sig.value_counts())

    # ------------------ Evaluate strategy ------------------
    ar_perf, ar_eq = evaluate_strategy(test, arima_sig)
    ar_perf["RMSE"] = math.sqrt(mean_squared_error(test["ret"], pred_returns))

    # ------------------ Store results ------------------
    bench_results[arima_name] = ar_perf
    bench_curves[arima_name] = ar_eq

except Exception as e:
    print(f"[WARN] ARIMA failed: {e}")

'''
# ---------------------- LSTM MODEL ----------------------
lstm_name = "LSTM"
if USE_LSTM:
    try:
        # Features: [ret, sma_ratio, rsi14, vol_10]
        FEATS = ["ret", "sma_ratio", "rsi14", "vol_10"]
        lookback = 20
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()

        feat_train = train[FEATS].copy()
        feat_test  = test[FEATS].copy()
        # Handle potential NaNs (should be none after add_indicators dropna)
        feat_train = feat_train.fillna(method="ffill").fillna(0)
        feat_test  = feat_test.fillna(method="ffill").fillna(0)

        X_all = pd.concat([feat_train, feat_test], axis=0)
        scaler_x.fit(X_all.values)
        Xs = scaler_x.transform(X_all.values)

        y_all = pd.concat([train["ret"], test["ret"]], axis=0).values.reshape(-1,1)
        scaler_y.fit(y_all[:len(train)])
        ys = scaler_y.transform(y_all)

        def make_sequences(X, y, start, end, lookback):
            Xseq, yseq, idx = [], [], []
            for i in range(start + lookback, end):
                Xseq.append(X[i-lookback:i])
                yseq.append(y[i])
                idx.append(i)
            return np.array(Xseq, dtype=np.float32), np.array(yseq, dtype=np.float32), idx

        # train set
        Xseq_tr, yseq_tr, _ = make_sequences(Xs, ys, 0, len(train), lookback)
        # test set aligned at the end
        Xseq_te, yseq_te, idx_te = make_sequences(Xs, ys, len(train), len(Xs), lookback)
        idx_te = np.array(idx_te)

        model_lstm = models.Sequential([
            layers.Input(shape=(lookback, len(FEATS))),
            layers.LSTM(32),
            layers.Dense(16, activation="relu"),
            layers.Dense(1, activation="linear")
        ])
        model_lstm.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
        es = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor="val_loss")
        hist = model_lstm.fit(
            Xseq_tr, yseq_tr, epochs=100, batch_size=32,
            validation_split=0.2, verbose=0, callbacks=[es]
        )

        yhat_te = model_lstm.predict(Xseq_te, verbose=0).reshape(-1,1)
        # Inverse transform predictions to returns
        yhat_te_inv = scaler_y.inverse_transform(yhat_te).flatten()

        # Map sequence predictions to the last EVAL_DAYS of test index
        pred_idx = pd.Index(X_all.index[idx_te])
        preds_series = pd.Series(yhat_te_inv, index=pred_idx)
        preds_series = preds_series.loc[test.index.intersection(pred_idx)]

        # ------------------ Generate long-only signals using percentile ------------------
        # Instead of >0, use top 45% predicted returns as long
        upper_pct = 0.6
        threshold = preds_series.quantile(upper_pct)
        lstm_sig = (preds_series >= threshold).astype(float)

        print("LSTM signal counts:\n", lstm_sig.value_counts())

        
        lstm_perf, lstm_eq = evaluate_strategy(test, lstm_sig)
        lstm_perf["RMSE"] = math.sqrt(mean_squared_error(test["ret"].reindex(preds_series.index), preds_series))
        bench_results[lstm_name] = lstm_perf
        bench_curves[lstm_name]  = lstm_eq
    except Exception as e:
        print(f"[WARN] LSTM failed: {e}")

# ---------------------- RL (DQN) ----------------------
rl_name = "RL_DQN"
if USE_RL:
    try:
        # Simple environment: state = window of tech features; action = {0: flat, 1: long}
        FEATS_RL = ["ret", "sma_ratio", "rsi14", "vol_10"]
        WIN = 10

        # Build arrays for RL period (train only for learning; evaluate on test after)
        feat_rl_tr = train[FEATS_RL].fillna(method="ffill").fillna(0).values
        ret_tr = train["ret"].values
        feat_rl_te = test[FEATS_RL].fillna(method="ffill").fillna(0).values
        ret_te = test["ret"].values

        sx = StandardScaler()
        feat_all = np.vstack([feat_rl_tr, feat_rl_te])
        sx.fit(feat_all)
        Xtr = sx.transform(feat_rl_tr)
        Xte = sx.transform(feat_rl_te)

        def build_states(X, win):
            S = []
            for i in range(win, len(X)):
                S.append(X[i-win:i].flatten())
            return np.array(S, dtype=np.float32)

        Str = build_states(Xtr, WIN)        # shape: [T_tr - WIN, WIN * F]
        Rtr = ret_tr[WIN:]                   # reward proxy (used to compute reward * action)
        Ste = build_states(Xte, WIN)
        Rte = ret_te[WIN:]

        state_dim = Str.shape[1]
        n_actions = 2  # 0: flat, 1: long

        class QNet(nn.Module):
            def __init__(self, inp, nA):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(inp, 64), nn.ReLU(),
                    nn.Linear(64, 32), nn.ReLU(),
                    nn.Linear(32, nA)
                )
            def forward(self, x):
                return self.net(x)

        qnet = QNet(state_dim, n_actions)
        target = QNet(state_dim, n_actions)
        target.load_state_dict(qnet.state_dict())

        optimizer = optim.Adam(qnet.parameters(), lr=1e-3)
        gamma = 0.95
        eps_start, eps_end, eps_decay_steps = 0.9, 0.05, 1000
        step_count = 0
        batch_size = 64
        memory = []
        MEM_CAP = 5000
        updates = 0

        def eps_greedy(qvals, eps):
            if np.random.rand() < eps:
                return np.random.randint(n_actions)
            return int(torch.argmax(qvals).item())

        def sample_batch():
            idx = np.random.choice(len(memory), size=min(batch_size, len(memory)), replace=False)
            s,a,r,ns,d = zip(*[memory[i] for i in idx])
            return (torch.tensor(np.array(s), dtype=torch.float32),
                    torch.tensor(np.array(a), dtype=torch.int64).view(-1,1),
                    torch.tensor(np.array(r), dtype=torch.float32).view(-1,1),
                    torch.tensor(np.array(ns), dtype=torch.float32),
                    torch.tensor(np.array(d), dtype=torch.float32).view(-1,1))

        # Train on training window with multiple passes (epochs) through sequence
        EPOCHS = 8
        TARGET_SYNC = 200

        for epoch in range(EPOCHS):
            pos = 0  # previous action (for cost calc)
            for t in range(len(Str)-1):
                s = torch.tensor(Str[t], dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    qvals = qnet(s)
                eps = eps_end + (eps_start - eps_end) * math.exp(-step_count / eps_decay_steps)
                a = eps_greedy(qvals, eps)

                # reward = next day's return * position  - cost if position changes
                next_ret = Rtr[t+0]   # reward for taking action at t and holding for next day
                cost = (TXN_COST_BPS/1e4) * (1.0 if a != pos else 0.0)
                r = (next_ret * (1 if a==1 else 0)) - cost
                pos = a

                ns = Str[t+1]
                done = 1.0 if (t+1 == len(Str)-1) else 0.0

                if len(memory) >= MEM_CAP:
                    memory.pop(0)
                memory.append((Str[t], a, r, ns, done))

                if len(memory) >= 200:
                    S,A,R,NS,D = sample_batch()
                    with torch.no_grad():
                        max_next = torch.max(target(NS), dim=1, keepdim=True)[0]
                        y = R + (1 - D) * gamma * max_next
                    q = qnet(S).gather(1, A)
                    loss = nn.functional.mse_loss(q, y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    updates += 1
                    if updates % TARGET_SYNC == 0:
                        target.load_state_dict(qnet.state_dict())
                step_count += 1

        # Evaluate on test set (greedy, no learning)
        pos = 0
        actions = []
        for t in range(len(Ste)-1):
            s = torch.tensor(Ste[t], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                a = int(torch.argmax(qnet(s)).item())
            actions.append(a)
            pos = a
        # Align actions to test index (skip initial WIN for state build and last step alignment)
        act_series = pd.Series(actions, index=test.index[WIN+1:WIN+1+len(actions)])
        rl_sig = act_series.reindex(test.index).fillna(method="ffill").fillna(0.0).clip(0,1)

        rl_perf, rl_eq = evaluate_strategy(test, rl_sig)
        rl_perf["RMSE"] = np.nan  # RL is not a forecaster
        bench_results[rl_name] = rl_perf
        bench_curves[rl_name]  = rl_eq

    except Exception as e:
        print(f"[WARN] RL failed: {e}")

# ---------------------- RESULTS ----------------------
# Plot equity curves
plot_equity_curves(bench_curves, "Equity Curves – Test (last ~6 weeks)", os.path.join(OUTDIR, "equity_curves_test.png"))

# Model diagnostics plots (optional: ARIMA & LSTM preds vs actual)
try:
    if "ARIMA" in bench_results:
        plt.figure(figsize=(10,4))
        plt.plot(test.index, test["ret"], label="Actual ret")
        plt.plot(test.index, preds, label="ARIMA pred")
        plt.title("ARIMA: predicted daily returns (test)")
        plt.legend(); plt.grid(True); plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, "arima_preds.png"), dpi=130)
        plt.close()
except:
    pass

try:
    if USE_LSTM and "LSTM" in bench_results:
        plt.figure(figsize=(10,4))
        # preds_series defined above in LSTM block
        aligned = test["ret"].reindex(preds_series.index)
        plt.plot(aligned.index, aligned.values, label="Actual ret")
        plt.plot(preds_series.index, preds_series.values, label="LSTM pred")
        plt.title("LSTM: predicted daily returns (test)")
        plt.legend(); plt.grid(True); plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, "lstm_preds.png"), dpi=130)
        plt.close()
except:
    pass

# Leaderboard
# Add RMSE placeholder for benchmarks
for k in ["BuyHold", "SMA_5over20"]:
    if "RMSE" not in bench_results[k]:
        bench_results[k]["RMSE"] = np.nan

leader = print_leaderboard(bench_results)

# Save table
leader.to_csv(os.path.join(OUTDIR, "performance_table.csv"))

print(f"\nSaved figures & table in: {OUTDIR}/")
