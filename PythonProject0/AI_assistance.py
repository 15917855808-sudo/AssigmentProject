"""
Module 4: Natural Language Q&A CLI for NYC Taxi Analysis
- Uses OpenRouter LLM to parse user questions into {intent, params}
- Dispatches to handlers that call functions from M1-M3
- Returns numeric answer + chart path
"""

import os
import re
import json
import time
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_pipeline      import run_pipeline                # M1
from picture_generation import (analyze_time_pattern,      # M2
                                analyze_zone_hotspot,
                                analyze_fare_factors,
                                analyze_speed_period)
from data_prediction    import build_demand_table          # M3

from sklearn.ensemble       import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ============ Config ============
OPENROUTER_API_KEY = "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"
OPENROUTER_URL     = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME         = "openai/gpt-4o-mini"   # cheap & supports JSON

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)


# ============ LLM Intent Parsing ============
SYSTEM_PROMPT = """You are an intent parser for a NYC Taxi data assistant.
Read the user's question (Chinese or English) and return STRICT JSON only:

{
  "intent": "<one of: hourly_demand | zone_ranking | demand_predict | fare_estimate | speed_query | overview | unknown>",
  "params": { ... }
}

Intent definitions and params:
- hourly_demand: query trip volume by hour / weekday.
    params: {"hour": int|null, "weekday": int|null (0=Mon..6=Sun), "is_weekend": 0|1|null}
- zone_ranking: top-N pickup or dropoff zones.
    params: {"kind": "pickup"|"dropoff", "top_n": int (default 10)}
- demand_predict: predict demand for a zone at a specific hour/weekday.
    params: {"zone_id": int, "hour": int, "weekday": int}
- fare_estimate: estimate fare given distance / period / passengers.
    params: {"distance": float, "is_rush_hour": 0|1|null, "passenger_count": int|null}
- speed_query: average trip speed at certain hour/period.
    params: {"hour": int|null, "is_rush_hour": 0|1|null}
- overview: regenerate full analytic charts.
    params: {}
- unknown: anything else.

Output ONLY the JSON object, no extra text.
"""

def llm_parse(question: str) -> dict:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": question},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    try:
        r = requests.post(OPENROUTER_URL, headers=headers,
                          json=payload, timeout=30)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        return json.loads(content)
    except Exception as e:
        print(f"[LLM error] {e}, falling back to rule-based parser")
        return rule_based_parse(question)


def rule_based_parse(q: str) -> dict:
    """Fallback: simple regex parser when API fails."""
    ql = q.lower()
    if any(w in ql for w in ["predict", "预测", "需求量"]):
        zone = re.search(r"(?:zone|区域)\s*(\d+)", ql)
        hour = re.search(r"(\d{1,2})\s*(?:点|h|:00|时)", ql)
        return {"intent": "demand_predict",
                "params": {"zone_id": int(zone.group(1)) if zone else 132,
                           "hour": int(hour.group(1)) if hour else 8,
                           "weekday": 0}}
    if any(w in ql for w in ["fare", "费用", "车费", "多少钱"]):
        d = re.search(r"(\d+(?:\.\d+)?)\s*(?:mile|英里|公里|km)", ql)
        return {"intent": "fare_estimate",
                "params": {"distance": float(d.group(1)) if d else 5.0,
                           "is_rush_hour": 1 if "高峰" in ql or "rush" in ql else 0,
                           "passenger_count": 1}}
    if any(w in ql for w in ["top", "排名", "热门", "最多"]):
        return {"intent": "zone_ranking",
                "params": {"kind": "dropoff" if "下客" in ql or "drop" in ql else "pickup",
                           "top_n": 10}}
    if any(w in ql for w in ["speed", "速度"]):
        return {"intent": "speed_query", "params": {"is_rush_hour": None}}
    if any(w in ql for w in ["hour", "小时", "几点", "时段"]):
        return {"intent": "hourly_demand", "params": {}}
    if any(w in ql for w in ["overview", "总览", "全部"]):
        return {"intent": "overview", "params": {}}
    return {"intent": "unknown", "params": {}}


# ============ Handlers ============
def _save_fig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return path


def handle_hourly_demand(df, p):
    hour      = p.get("hour")
    weekday   = p.get("weekday")
    is_wknd   = p.get("is_weekend")

    sub = df.copy()
    if is_wknd is not None:
        sub = sub[sub["is_weekend"] == is_wknd]
    if weekday is not None:
        sub = sub[sub["pickup_weekday"] == weekday]

    days = sub["tpep_pickup_datetime"].dt.date.nunique() or 1
    hourly = sub.groupby("pickup_hour").size() / days

    if hour is not None and hour in hourly.index:
        ans = f"Avg trips at hour {hour}: {hourly[hour]:.1f}"
    else:
        peak_h = int(hourly.idxmax())
        ans = f"Peak hour = {peak_h} ({hourly.max():.1f} trips/day avg)"

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(hourly.index, hourly.values, marker="o", color="steelblue")
    if hour is not None:
        ax.axvline(hour, color="red", ls="--", alpha=0.6)
    ax.set(xlabel="Hour of Day", ylabel="Avg Trips / Day",
           xticks=range(24), title="Hourly Demand (Filtered)")
    path = _save_fig(fig, f"qa_hourly_{int(time.time())}.png")
    return ans, path


def handle_zone_ranking(df, p):
    kind  = p.get("kind", "pickup")
    top_n = int(p.get("top_n", 10))
    col   = "PULocationID" if kind == "pickup" else "DOLocationID"
    top   = df[col].value_counts().head(top_n)

    ans = f"Top {top_n} {kind} zones: " + \
          ", ".join(f"{z}({c})" for z, c in top.items())

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(top.index.astype(str)[::-1], top.values[::-1], color="teal")
    ax.set(xlabel="Trip Count", ylabel=f"{kind.title()} Zone ID",
           title=f"Top {top_n} {kind.title()} Zones")
    path = _save_fig(fig, f"qa_zones_{kind}_{int(time.time())}.png")
    return ans, path


# Cache RF model so prediction is fast
_RF_CACHE = {"model": None, "feat_cols": None, "demand": None}

def _get_rf(df):
    if _RF_CACHE["model"] is not None:
        return _RF_CACHE
    demand = build_demand_table(df)
    feat_cols = ["PULocationID", "pickup_hour", "weekday",
                 "is_weekend", "is_rush_hour",
                 "hour_sin", "hour_cos", "wday_sin", "wday_cos"]
    X, y = demand[feat_cols].values, demand["demand"].values
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=120, n_jobs=-1, random_state=42)
    rf.fit(X_tr, y_tr)
    _RF_CACHE.update(model=rf, feat_cols=feat_cols, demand=demand)
    print("[info] RF model trained & cached")
    return _RF_CACHE


def handle_demand_predict(df, p):
    cache = _get_rf(df)
    rf, demand = cache["model"], cache["demand"]
    zid  = int(p.get("zone_id", 132))
    hr   = int(p.get("hour", 8))
    wd   = int(p.get("weekday", 0))

    if zid not in demand["PULocationID"].unique():
        zid = int(demand["PULocationID"].mode()[0])

    is_wknd = int(wd >= 5)
    is_rush = int((not is_wknd) and hr in [7, 8, 9, 17, 18, 19])
    feats = np.array([[zid, hr, wd, is_wknd, is_rush,
                       np.sin(2*np.pi*hr/24), np.cos(2*np.pi*hr/24),
                       np.sin(2*np.pi*wd/7),  np.cos(2*np.pi*wd/7)]])
    pred = float(rf.predict(feats)[0])
    ans  = f"Predicted demand for zone {zid}, weekday {wd}, hour {hr}: ~{pred:.1f} trips"

    # Compare with historical avg in that zone
    hist = (demand[(demand["PULocationID"] == zid) &
                   (demand["pickup_hour"] == hr)]
                  .groupby("weekday")["demand"].mean())
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(hist.index, hist.values, color="lightgray", label="Historical avg")
    ax.axhline(pred, color="red", ls="--", label=f"Predicted ({pred:.1f})")
    ax.scatter([wd], [pred], color="red", zorder=5, s=80)
    ax.set(xlabel="Weekday (0=Mon)", ylabel="Demand",
           title=f"Zone {zid} @ Hour {hr}: Predicted vs Historical",
           xticks=range(7))
    ax.legend()
    path = _save_fig(fig, f"qa_predict_{zid}_{hr}_{int(time.time())}.png")
    return ans, path


def handle_fare_estimate(df, p):
    dist     = float(p.get("distance", 5.0))
    is_rush  = p.get("is_rush_hour")
    pcount   = p.get("passenger_count")

    sub = df.copy()
    if is_rush is not None:
        sub = sub[sub["is_rush_hour"] == int(is_rush)]
    if pcount is not None:
        sub = sub[sub["passenger_count"] == int(pcount)]
    # Linear estimate: fare = a + b*distance (robust median fit)
    sub = sub[(sub["trip_distance"] > 0) & (sub["fare_amount"] > 0)]
    rate = (sub["fare_amount"] / sub["trip_distance"]).median()
    base = sub["fare_amount"].median() - rate * sub["trip_distance"].median()
    est  = base + rate * dist
    ans  = (f"Estimated fare for {dist} mi"
            f"{' (rush hour)' if is_rush == 1 else ''}: ${est:.2f} "
            f"(rate≈${rate:.2f}/mi, base≈${base:.2f})")

    # Plot scatter + estimate point
    s = sub.sample(min(20000, len(sub)), random_state=0)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(s["trip_distance"], s["fare_amount"], alpha=0.2, s=6, color="gray")
    xs = np.linspace(0, max(30, dist+5), 50)
    ax.plot(xs, base + rate * xs, "b-", label=f"Fit: {rate:.2f}x+{base:.2f}")
    ax.scatter([dist], [est], color="red", s=100, zorder=5,
               label=f"Estimate (${est:.2f})")
    ax.set(xlim=(0, 30), ylim=(0, 150),
           xlabel="Distance (miles)", ylabel="Fare ($)",
           title="Fare Estimate")
    ax.legend()
    path = _save_fig(fig, f"qa_fare_{int(time.time())}.png")
    return ans, path


def handle_speed_query(df, p):
    sub = df[df["avg_speed_mph"].between(1, 60)].copy()
    is_rush = p.get("is_rush_hour")
    hour    = p.get("hour")
    if is_rush is not None:
        sub = sub[sub["is_rush_hour"] == int(is_rush)]
    if hour is not None:
        sub = sub[sub["pickup_hour"] == int(hour)]

    avg = sub["avg_speed_mph"].mean()
    ans = f"Average speed under given conditions: {avg:.2f} mph"

    hourly = df[df["avg_speed_mph"].between(1, 60)] \
                .groupby("pickup_hour")["avg_speed_mph"].mean()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(hourly.index, hourly.values, marker="o", color="purple")
    if hour is not None:
        ax.axvline(hour, color="red", ls="--")
    ax.set(xlabel="Hour", ylabel="Avg Speed (mph)", xticks=range(24),
           title="Avg Speed by Hour")
    path = _save_fig(fig, f"qa_speed_{int(time.time())}.png")
    return ans, path


def handle_overview(df, p):
    analyze_time_pattern(df)
    analyze_zone_hotspot(df)
    analyze_fare_factors(df)
    analyze_speed_period(df)
    return "All analytic charts regenerated.", f"./{OUT_DIR}/"


HANDLERS = {
    "hourly_demand":  handle_hourly_demand,
    "zone_ranking":   handle_zone_ranking,
    "demand_predict": handle_demand_predict,
    "fare_estimate":  handle_fare_estimate,
    "speed_query":    handle_speed_query,
    "overview":       handle_overview,
}


# ============ CLI Loop ============
HELP_TEXT = """
Supported question examples:
  1. Hourly demand : "周五下午6点平均订单量" / "what's the peak hour?"
  2. Zone ranking  : "Top 10 pickup zones" / "下客量最高的5个区域"
  3. Demand predict: "predict demand for zone 132 at 8am Monday"
  4. Fare estimate : "10英里在晚高峰大概多少钱"
  5. Speed query   : "rush hour average speed" / "凌晨3点平均速度"
  6. Overview      : "regenerate all charts"
Type 'help' for this list, 'quit' to exit.
"""

def main():
    print(">>> Loading data ...")
    df = run_pipeline("yellow_tripdata_2026-01.parquet")
    print(">>> Ready.\n" + HELP_TEXT)

    while True:
        try:
            q = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye"); break
        if not q:
            continue
        if q.lower() in {"quit", "exit", "q"}:
            print("bye"); break
        if q.lower() == "help":
            print(HELP_TEXT); continue

        parsed = llm_parse(q)
        intent = parsed.get("intent", "unknown")
        params = parsed.get("params", {})
        print(f"[parsed] intent={intent}  params={params}")

        if intent not in HANDLERS:
            print("Bot> Sorry, I don't understand. Type 'help'.\n"); continue
        try:
            answer, chart = HANDLERS[intent](df, params)
            print(f"Bot> {answer}\n     chart: {chart}\n")
        except Exception as e:
            print(f"Bot> Error while answering: {e}\n")


if __name__ == "__main__":
    main()