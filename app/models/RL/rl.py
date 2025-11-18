# rl_nudge_excess_gymnasium.py
import os
import random
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import gymnasium as gym
from gymnasium import spaces

# Stable Baselines 3
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
except Exception:
    raise RuntimeError("Please install stable-baselines3: pip install stable-baselines3[extra]")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

CSV_PATH = "classification_data/synthetic_trade_sequences_planned.csv"
MAX_HISTORY_TRADES = 10
ARCHES = ["revenge","fomo","herd","disciplined","overconfident"]
NUDGES = ["dont_hesitate","calm_down","take_a_second","alert_stop"]
NUM_ACTIONS = len(NUDGES)


# ---------- small CSV generator (if missing) ----------
def generate_small_csv(path, n_sessions=1000):
    ARCHETYPES = ARCHES
    rows = []
    seq_id = 0
    for _ in range(n_sessions):
        seq_id += 1
        archetype = random.choice(ARCHETYPES)
        planned_max = {"revenge":20,"fomo":18,"herd":12,"disciplined":6,"overconfident":22}[archetype]
        actual_len = max(1, int(np.round(np.random.normal(planned_max * (0.9 if archetype=="disciplined" else 1.1), scale=max(1, planned_max*0.4)))))
        actual_len = min(max(actual_len, 1), planned_max + 30)
        start_price = round(np.random.uniform(20, 400),2)
        prices = [start_price]
        for i in range(actual_len+1):
            prices.append(max(0.01, prices[-1] * (1 + np.random.normal(0, 0.02))))
        last_ts = datetime(2025,1,1) + timedelta(days=random.randint(0,365))
        for i in range(actual_len):
            price = round(prices[i],2)
            next_price = round(prices[i+1],2)
            side = "BUY" if random.random() < 0.55 else "SELL"
            qty = int(np.random.choice([5,10,20]) * np.random.uniform(0.7,2.5))
            pnl = round((next_price - price) * (1 if side=="BUY" else -1) * qty, 2)
            time_delta = int(np.random.exponential(scale=600))
            last_ts = last_ts + timedelta(seconds=time_delta)
            rows.append({
                "sequence_id": seq_id,
                "archetype": archetype,
                "trade_idx": i+1,
                "timestamp": last_ts.isoformat(),
                "stock": random.choice(["AAPL","TSLA","NVDA","AMZN"]),
                "side": side,
                "qty": qty,
                "entry_price": price,
                "exit_price": next_price,
                "pnl": pnl,
                "position_size": round(qty*price,2),
                "leverage": round(np.random.uniform(1.0,3.0),2),
                "time_since_prev_sec": time_delta,
                "planned_max_trades": planned_max,
                "actual_trades_in_session": actual_len,
                "label": archetype
            })
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"Generated {len(df)} rows to {path}")
    return df

# ---------- load data & build sessions (filter only sessions with excess trades) ----------
if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH, parse_dates=["timestamp"])
else:
    df = generate_small_csv(CSV_PATH, n_sessions=1200)

def build_sessions_with_excess(df):
    sessions = []
    for seq_id, g in df.groupby("sequence_id"):
        g = g.sort_values("trade_idx")
        planned = int(g["planned_max_trades"].iloc[0])
        actual = int(g["actual_trades_in_session"].iloc[0]) if "actual_trades_in_session" in g.columns else int(g.shape[0])
        if actual <= planned:
            continue
        excess_df = g[g["trade_idx"] > planned].copy().reset_index(drop=True)
        baseline_excess_pnl = float(excess_df["pnl"].sum())
        before_df = g[g["trade_idx"] <= planned].copy()
        history_df = before_df.tail(MAX_HISTORY_TRADES) if before_df.shape[0] > 0 else g.iloc[:MAX_HISTORY_TRADES]
        sessions.append({
            "sequence_id": int(seq_id),
            "archetype": g["label"].iloc[0],
            "planned_max_trades": planned,
            "actual_trades": actual,
            "history_df": history_df.reset_index(drop=True),
            "excess_df": excess_df,
            "baseline_excess_pnl": baseline_excess_pnl
        })
    return sessions

sessions = build_sessions_with_excess(df)
print(f"Prepared {len(sessions)} sessions with excess trades (usable for RL).")


# ---------- action semantics ----------
def apply_nudge_to_excess(excess_df, nudge):
    L = len(excess_df)
    if L == 0:
        return excess_df.iloc[0:0]
    if nudge == "dont_hesitate":
        return excess_df.copy()
    if nudge == "alert_stop":
        return excess_df.iloc[0:0]
    if nudge == "calm_down":
        keep_k = max(1, int(math.ceil(L * 0.5)))
        idxs = list(range(L))
        keep = set(sorted(random.sample(idxs, keep_k)))
        return excess_df.iloc[sorted(list(keep))].reset_index(drop=True)
    if nudge == "take_a_second":
        if random.random() < 0.5:
            return excess_df.iloc[0:0]
        else:
            keep_k = max(1, int(math.ceil(L * 0.5)))
            idxs = list(range(L))
            keep = set(sorted(random.sample(idxs, keep_k)))
            return excess_df.iloc[sorted(list(keep))].reset_index(drop=True)
    return excess_df.copy()


# ---------- Environment (gymnasium-compatible) ----------
class NudgeExcessEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, sessions, max_history=MAX_HISTORY_TRADES):
        super().__init__()
        self.sessions = sessions
        self.max_history = max_history
        self.per_trade_features = 5
        self.n_arche = len(ARCHES)
        self.obs_dim = self.max_history * self.per_trade_features + 5 + self.n_arche
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.curr = None

    # Accept seed (SB3 passes seed=) and options (gymnasium API)
    # replace the existing reset and step methods in NudgeExcessEnv with this

    # Accept seed (SB3 calls env.reset(seed=...))
    def reset(self, *, seed=None, options=None):
        # seed RNGs for reproducibility when requested
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        # choose a random session
        self.curr = random.choice(self.sessions)
        self.baseline_excess_pnl = self.curr["baseline_excess_pnl"]
        obs = self._make_obs(self.curr)
        info = {}  # you can include metadata here if needed
        # Gymnasium expects (obs, info)
        return obs.astype(np.float32), info

    def step(self, action):
        assert self.curr is not None, "call reset() before step()"
        nudge = NUDGES[int(action)]
        excess_df = self.curr["excess_df"]
        kept = apply_nudge_to_excess(excess_df, nudge)
        new_excess_pnl = float(kept["pnl"].sum()) if kept.shape[0] > 0 else 0.0
        reward = float(new_excess_pnl - self.baseline_excess_pnl)

        info = {
            "sequence_id": self.curr["sequence_id"],
            "archetype": self.curr["archetype"],
            "planned_max_trades": self.curr["planned_max_trades"],
            "actual_trades": self.curr["actual_trades"],
            "nudge": nudge,
            "baseline_excess_pnl": self.baseline_excess_pnl,
            "new_excess_pnl": new_excess_pnl,
            "kept_excess_trades": int(kept.shape[0])
        }

        # single-step episode: terminated=True, truncated=False (gymnasium style)
        terminated = True
        truncated = False

        obs = self._make_obs(self.curr)
        # Gymnasium expects (obs, reward, terminated, truncated, info)
        return obs.astype(np.float32), float(reward), terminated, truncated, info


    def _make_obs(self, sess):
        hist = sess["history_df"]
        L = hist.shape[0]
        trade_feats = []
        for i in range(max(0, L - self.max_history), L):
            row = hist.iloc[i]
            side = 1.0 if str(row["side"]).upper() == "BUY" else 0.0
            trade_feats.append([side,
                                float(row.get("qty", 0.0)),
                                float(row.get("entry_price", 0.0)),
                                float(row.get("pnl", 0.0)),
                                float(row.get("time_since_prev_sec", 0.0))])
        while len(trade_feats) < self.max_history:
            trade_feats.insert(0, [0.0]*self.per_trade_features)
        trade_feats = np.array(trade_feats, dtype=np.float32).reshape(-1)
        planned = float(sess["planned_max_trades"])
        actual = float(sess["actual_trades"])
        last_pnl = float(sess["history_df"]["pnl"].iloc[-1]) if sess["history_df"].shape[0] > 0 else 0.0
        avg_pnl = float(sess["history_df"]["pnl"].mean()) if sess["history_df"].shape[0] > 0 else 0.0
        win_rate = float((sess["history_df"]["pnl"] > 0).sum() / max(1, sess["history_df"].shape[0]))
        planned_norm = planned / (planned + 1.0)
        actual_norm = actual / (actual + 1.0)
        sess_feats = np.array([planned_norm, actual_norm, last_pnl, avg_pnl, win_rate], dtype=np.float32)
        arche_onehot = np.array([1.0 if sess["archetype"] == a else 0.0 for a in ARCHES], dtype=np.float32)
        obs = np.concatenate([trade_feats, sess_feats, arche_onehot], axis=0)
        return obs


# ---------- training helper (fix net_arch format to dict) ----------
def train_agent(env, timesteps=100000):
    venv = DummyVecEnv([lambda: env])
    policy_kwargs = dict(net_arch=dict(pi=[128,64], vf=[128,64]))
    model = PPO("MlpPolicy", venv, verbose=1, policy_kwargs=policy_kwargs)
    model.learn(total_timesteps=timesteps)
    return model


# ---------- run training ----------
env = NudgeExcessEnv(sessions)
print("Prepared env with obs_dim =", env.observation_space.shape, "action_space =", env.action_space)
model = train_agent(env, timesteps=100000)

# ---------- evaluate policy ----------
import numpy as np

def evaluate(model, env, n=200):
    """
    Run deterministic policy on env n times. Works with gymnasium-style env.reset()/step().
    Returns list of info dicts for each episode.
    """
    results = []
    for _ in range(n):
        # gymnasium reset returns (obs, info)
        maybe = env.reset()
        if isinstance(maybe, tuple) and len(maybe) == 2:
            obs, reset_info = maybe
        else:
            # fallback for older gym-style envs
            obs = maybe
            reset_info = {}

        # ensure obs is numpy array (SB3 expects array/dict)
        if not isinstance(obs, np.ndarray):
            obs = np.asarray(obs, dtype=np.float32)

        # model.predict expects just the observation (not (obs,info))
        action, _states = model.predict(obs, deterministic=True)

        # model.predict may return array-like (e.g. np.array([a])) or scalar
        # get a single int action
        if isinstance(action, (list, tuple, np.ndarray)):
            action = int(np.asarray(action).flatten()[0])
        else:
            action = int(action)

        # gymnasium step returns (obs, reward, terminated, truncated, info)
        step_res = env.step(action)
        if len(step_res) == 5:
            obs2, reward, terminated, truncated, info = step_res
            done = bool(terminated or truncated)
        else:
            # fallback to gym-style (obs, reward, done, info)
            obs2, reward, done, info = step_res
            terminated = done
            truncated = False

        results.append(info)
    return results


res = evaluate(model, env, n=200)
from collections import Counter
ctr = Counter([r["nudge"] for r in res])
print("Nudge counts (sample):", ctr)
avg_reward = np.mean([r["new_excess_pnl"] - r["baseline_excess_pnl"] for r in res])
print("Average reward over eval:", avg_reward)
for i, r in enumerate(res[:10]):
    print(i, r)
model.save("ppo_nudge_excess_model_gymnasium")
print("Saved model ppo_nudge_excess_model_gymnasium")
