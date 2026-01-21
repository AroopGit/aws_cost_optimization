"""
FastAPI backend for GCPL Pricing Simulator
Implements the full hybrid pricing logic from gcpl_pricing_hybrid_full.py
"""

import os
import sys
import json
import math
import random
import re
import io
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
import uvicorn

# ML and Forecasting imports
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.arima.model import ARIMA
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("Warning: ML libraries not available. Install scikit-learn and statsmodels for forecasting features.")

# Bedrock (boto3)
try:
    import boto3
except ImportError:
    boto3 = None

app = FastAPI(
    title="Pricing Simulator API",
    description="Hybrid pricing simulator with 3-variant agents: Base -> Promo -> Competitor -> Inventory -> Selection",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# CONFIG
# -----------------------------
DATA_CSV = os.getenv("DATA_CSV", "./data/synthetic_detailed_v2_named.csv")
OUTPUT_DIR = "./gcpl_pricing_outputs"

# -----------------------------
# DEFAULT GUARDRAILS
# -----------------------------
DEFAULT_GUARDRAILS = {
    "min_margin_pct": 12.0,
    "max_promo_depth_pct": 40.0,
    "auto_approve_pct": 2.0,
    "manager_review_pct": 5.0,
    "max_monthly_change_pct": 15.0
}

# Global guardrails
current_guardrails = DEFAULT_GUARDRAILS.copy()

# Global dataset
dataset_df = None

# -----------------------------
# Utility helpers
# -----------------------------
def ensure_output_dir(path=OUTPUT_DIR):
    Path(path).mkdir(parents=True, exist_ok=True)
    return path

def safe_float(x, default=np.nan):
    try:
        if pd.isna(x):
            return default
    except Exception:
        pass
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default

def pct_change(a, b):
    try:
        a_f, b_f = float(a), float(b)
    except Exception:
        return 0.0
    if a_f == 0.0:
        return 0.0 if b_f == 0.0 else 100.0
    return (b_f - a_f) / (abs(a_f) + 1e-9) * 100.0

    return (b_f - a_f) / (abs(a_f) + 1e-9) * 100.0

# -----------------------------
# NL Parser Helpers
# -----------------------------
def parse_nl_scenario(text: str) -> Dict[str, Any]:
    """
    Improved NL parser:
      - Detects local price, competitor price (including phrases like 'competitor drops to 290'),
      - Recognizes promo depth (-0.03 or -3% or 3%),
      - Inventory days,
      - Channel mentions and SKU mentions.
    Returns overrides dict (keys: local_price, competitor_price, promo_depth_pct, inventory_days, channel, sku_id).
    """
    text_raw = text
    text = text.lower()
    overrides = {}
    num_re = r"([0-9]+(?:\.[0-9]+)?)"

    # competitor patterns: 'competitor drops to 290', 'competitor to 290', 'competitor = 290', 'comp drops to 290'
    m = re.search(r"(?:competitor|comp|rival|peer)\s*(?:drops\s*to|down\s*to|to|=|is|becomes)\s*₹?\s*"+num_re, text)
    if m:
        overrides["competitor_price"] = float(m.group(1))

    # competitor patterns like 'drops by 10%' => relative
    m = re.search(r"(?:competitor|comp)\s*(?:drops|down)\s*(?:by)?\s*"+num_re+r"\s*%+", text)
    if m and "competitor_price" not in overrides:
        try:
            pct = float(m.group(1))
            # can't compute absolute price without baseline; leave a special flag to indicate relative competitor drop
            overrides["competitor_pct_change"] = -abs(pct)
        except Exception:
            pass

    # local price
    m = re.search(r"(?:local price|my price|our price|price)\s*(?:is|=|at|to|becomes)?\s*₹?\s*"+num_re, text)
    if m:
        overrides["local_price"] = float(m.group(1))

    # promo depth: 'promo -0.03', 'promo -3%', 'promo 3%'
    m = re.search(r"(?:promo|promotion|discount|promo depth)\s*(?:is|=|at)?\s*([+-]?[0-9]+(?:\.[0-9]+)?)\s*%?", text)
    if m:
        val = float(m.group(1))
        if abs(val) < 1 and "." in m.group(1):
            val = val * 100.0
        overrides["promo_depth_pct"] = float(val)

    # inventory days
    m = re.search(r"(?:inventory days|inventory)\s*(?:is|=|at|:)?\s*"+num_re, text)
    if m:
        overrides["inventory_days"] = float(m.group(1))

    # channel
    if re.search(r"\becom\b|\be-commerce\b|\bonline\b", text):
        overrides["channel"] = "ECOM"
    elif re.search(r"\bgt\b|\bgeneral trade\b", text):
        overrides["channel"] = "GT"
    elif re.search(r"\bmt\b|\bmodern trade\b", text):
        overrides["channel"] = "MT"

    # SKU patterns (SKU_0001, GCPL_SKU_0001, sku 0001)
    m = re.search(r"(sku[_\-\s]?[0-9a-zA-Z]+)", text, flags=re.I)
    if m:
        overrides["sku_id"] = m.group(1).upper().replace(" ", "_")

    # fallback: detect plain numbers that appear near 'competitor' words with different phrasing
    if "competitor_price" not in overrides:
        m2 = re.search(r"competitor.*?([0-9]+(?:\.[0-9]+)?)", text)
        if m2:
            overrides["competitor_price"] = float(m2.group(1))

    return overrides

def bedrock_invoke(messages: list, model_id: str = "openai.gpt-oss-20b-1:0", max_tokens: int = 600, temperature: float = 0.0):
    if boto3 is None:
        raise RuntimeError("boto3 not installed/configured to call Bedrock.")
    client = boto3.client("bedrock-runtime", region_name="us-east-1")
    native_request = {"model": model_id, "messages": messages, "max_completion_tokens": max_tokens, "temperature": temperature, "top_p": 0.9, "stream": False}
    import json as _json
    response = client.invoke_model(modelId=model_id, body=_json.dumps(native_request))
    body = response['body'].read().decode('utf-8')
    parsed = _json.loads(body)
    out_text = ""
    try:
        for choice in parsed.get("choices", []):
            msg = choice.get("message", {}).get("content")
            if msg:
                out_text += msg
    except Exception:
        out_text = parsed.get("output", parsed.get("text", str(parsed)))
    return out_text

def nl_interpret_with_bedrock(nl_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """
    If Bedrock configured, ask the LLM to return a JSON with overrides.
    """
    parsed = parse_nl_scenario(nl_text)
    if boto3:
        try:
            system = {
                "role": "system",
                "content": (
                    "You are a strict extractor. Given a short natural-language pricing scenario, "
                    "output *ONLY* a single JSON object (no surrounding text) with keys: overrides, extraction_confidence, notes. "
                    "The 'overrides' value must be a JSON object containing any of: local_price (number), competitor_price (number), "
                    "promo_depth_pct (number, percent), inventory_days (number), channel (ECOM|GT|MT), sku_id (string). "
                    "Do NOT propose business recommendations or new prices. If uncertain, set extraction_confidence to 'low' and include best-guess values in overrides."
                )
            }
            user = {
                "role": "user",
                "content": f"Scenario: {nl_text}\nContext (optional): {json.dumps(context)[:1200]}"
            }
            messages = [system, user]
            out = bedrock_invoke(messages, max_tokens=250, temperature=0.0)
            # try to find the first JSON object in the output
            jtxt = None
            try:
                # locate first '{' and last '}' that make a valid JSON
                first = out.find('{')
                last = out.rfind('}')
                if first != -1 and last != -1:
                    jtxt = out[first:last+1]
                    obj = json.loads(jtxt)
                    # validate shape
                    if "overrides" in obj and isinstance(obj["overrides"], dict):
                        # coerce numeric strings to numbers where possible
                        cleaned = {}
                        for k, v in obj["overrides"].items():
                            try:
                                if isinstance(v, (int, float)):
                                    cleaned[k] = float(v)
                                elif isinstance(v, str) and re.match(r"^-?\d+(\.\d+)?$", v.strip()):
                                    cleaned[k] = float(v.strip())
                                else:
                                    cleaned[k] = v
                            except Exception:
                                cleaned[k] = v
                        obj["overrides"] = cleaned
                        return obj
            except Exception:
                pass
            # if bedrock output invalid, fallback to parser
            return {"overrides": parsed, "extraction_confidence": "low", "notes": "Bedrock output invalid or could not parse; falling back to rule-based parser."}
        except Exception as e:
            print(f"Bedrock interpretation failed: {e}")
            return {"overrides": parsed, "extraction_confidence": "low", "notes": f"Bedrock error: {e}. Used rule-based parser."}
    else:
        return {"overrides": parsed, "extraction_confidence": "medium", "notes": "Bedrock unavailable; used local parser."}

# -----------------------------
# AgentTrace dataclass
# -----------------------------
@dataclass
class AgentTrace:
    name: str
    candidates: Dict[str, float] = field(default_factory=dict)  # conservative, neutral, aggressive
    details: Dict[str, Any] = field(default_factory=dict)

# -----------------------------
# Seasonality enrichment
# -----------------------------
def enrich_seasonality(df: pd.DataFrame) -> pd.DataFrame:
    """Compute monthly seasonality index from units_sold if missing or constant"""
    try:
        if "seasonality_index" not in df.columns or df["seasonality_index"].nunique(dropna=True) <= 1:
            if "date" in df.columns and "units_sold" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df["month"] = df["date"].dt.month
                monthly_avg = df.groupby("month")["units_sold"].mean().fillna(0.0)
                if monthly_avg.sum() > 0:
                    monthly_idx = (monthly_avg / monthly_avg.mean()).to_dict()
                    df["seasonality_index"] = df["month"].map(monthly_idx).fillna(1.0)
                else:
                    df["seasonality_index"] = 1.0
                df.drop(columns=["month"], inplace=True, errors=True)
            else:
                df["seasonality_index"] = 1.0
    except Exception as e:
        print(f"Seasonality enrichment failed: {e}")
        if "seasonality_index" not in df.columns:
            df["seasonality_index"] = 1.0
    return df

# -----------------------------
# Elasticity estimation
# -----------------------------
def estimate_elasticity(df: pd.DataFrame, sku: Optional[str] = None, category: Optional[str] = None) -> float:
    """Estimate price elasticity using log-log OLS"""
    def loglog_slope(x, y):
        try:
            xm = np.log1p(np.array(x).astype(float))
            ym = np.log1p(np.array(y).astype(float))
            A = np.vstack([xm, np.ones_like(xm)]).T
            sol, *_ = np.linalg.lstsq(A, ym, rcond=None)
            return float(sol[0])
        except Exception:
            return None
    
    if sku is not None:
        sdf = df[df["sku_id"].astype(str) == str(sku)]
        if len(sdf) >= 10:
            e = loglog_slope(
                sdf["local_price"].replace(0, np.nan).dropna(),
                sdf["units_sold"].replace(0, np.nan).dropna()
            )
            if e is not None and not math.isnan(e) and abs(e) > 0:
                return max(min(e, -0.1), -5.0)
    
    if category is not None:
        cdf = df[df["category"].astype(str) == str(category)]
        if len(cdf) >= 30:
            e = loglog_slope(
                cdf["local_price"].replace(0, np.nan).dropna(),
                cdf["units_sold"].replace(0, np.nan).dropna()
            )
            if e is not None and not math.isnan(e) and abs(e) > 0:
                return max(min(e, -0.1), -5.0)
    
    gdf = df.dropna(subset=["local_price", "units_sold"])
    if len(gdf) >= 50:
        e = loglog_slope(gdf["local_price"], gdf["units_sold"])
        if e is not None and not math.isnan(e) and abs(e) > 0:
            return max(min(e, -0.1), -5.0)
    
    return -0.8  # default

# -----------------------------
# Hybrid Agents (3 variants each)
# -----------------------------
class BasePriceAgent:
    def run(self, row: pd.Series, df_all: pd.DataFrame) -> AgentTrace:
        # Compute previous price from history: SKU median -> category-channel-region median -> global median
        prev_median = None
        sku = str(row.get("sku_id", ""))
        
        try:
            sku_hist = df_all[df_all["sku_id"].astype(str) == sku]
            if len(sku_hist) > 0:
                prev_median = float(sku_hist["local_price"].median(skipna=True))
        except Exception:
            prev_median = None
        
        if prev_median is None or math.isnan(prev_median):
            try:
                cat = str(row.get("category", ""))
                ch = str(row.get("channel", ""))
                rg = str(row.get("region", ""))
                grp = df_all[
                    (df_all["category"].astype(str) == cat) &
                    (df_all["channel"].astype(str) == ch) &
                    (df_all["region"].astype(str) == rg)
                ]
                if len(grp) > 0:
                    prev_median = float(grp["local_price"].median(skipna=True))
            except Exception:
                prev_median = None
        
        if prev_median is None or math.isnan(prev_median):
            prev_median = float(df_all["local_price"].median(skipna=True))
        
        seasonality = safe_float(row.get("seasonality_index"), 1.0)
        festival = safe_float(row.get("festival_lift"), 1.0)
        
        neutral = prev_median * seasonality * festival
        conservative = neutral * 0.98
        aggressive = neutral * 1.05
        
        return AgentTrace(
            "base",
            {
                "conservative": round(conservative, 2),
                "neutral": round(neutral, 2),
                "aggressive": round(aggressive, 2)
            },
            {"prev_median": prev_median, "seasonality": seasonality, "festival": festival}
        )

class PromoAgent:
    def __init__(self, guardrails: Dict[str, Any]):
        self.guard = guardrails
    
    def run(self, row: pd.Series, base_candidates: Dict[str, float]) -> AgentTrace:
        promo_pct_hist = safe_float(row.get("promo_depth_pct"), 0.0)
        max_p = self.guard.get("max_promo_depth_pct", 40.0)
        promo_pct_hist = min(max(promo_pct_hist, 0.0), max_p)
        
        cand = {}
        for lvl, base_price in base_candidates.items():
            if lvl == "conservative":
                cand[lvl] = base_price  # no promo
            elif lvl == "neutral":
                cand[lvl] = round(base_price * (1 - promo_pct_hist/100.0), 2)
            else:  # aggressive
                dp = min(promo_pct_hist + 10.0, max_p)
                cand[lvl] = round(base_price * (1 - dp/100.0), 2)
        
        return AgentTrace("promo", cand, {"applied_promo_pct": promo_pct_hist, "max_allowed": max_p})

class CompetitorAgent:
    def run(self, row: pd.Series, input_candidates: Dict[str, float]) -> AgentTrace:
        comp_price = safe_float(row.get("competitor_price"), None)
        out = {}
        
        if comp_price is None or math.isnan(comp_price):
            for lvl, p in input_candidates.items():
                out[lvl] = p
            return AgentTrace("competitor", out, {"note": "no competitor data"})
        
        for lvl, price_in in input_candidates.items():
            gap_pct = pct_change(price_in, comp_price)
            if lvl == "conservative":
                out[lvl] = price_in
            elif lvl == "neutral":
                if gap_pct < -5.0:
                    out[lvl] = round(price_in + 0.5*(comp_price - price_in), 2)
                elif gap_pct > 3.0:
                    out[lvl] = round(price_in * 1.02, 2)
                else:
                    out[lvl] = price_in
            else:  # aggressive
                if gap_pct < -5.0:
                    out[lvl] = round(price_in + 0.75*(comp_price - price_in), 2)
                elif gap_pct > 3.0:
                    out[lvl] = round(price_in * 1.03, 2)
                else:
                    out[lvl] = price_in
        
        return AgentTrace("competitor", out, {"comp_price": comp_price})

class InventoryAgent:
    def __init__(self, thresholds: Optional[Dict[str, Any]] = None):
        self.thresholds = thresholds or {"low": 10, "high": 60}
    
    def run(self, row: pd.Series, input_candidates: Dict[str, float]) -> AgentTrace:
        inv_days = None
        
        if "inventory_days" in row.index and not pd.isna(row.get("inventory_days")):
            inv_days = safe_float(row.get("inventory_days"), None)
        else:
            inv_level = safe_float(row.get("inventory_level"), None)
            units_sold = safe_float(row.get("units_sold"), None)
            
            if inv_level is not None and units_sold is not None and units_sold > 0:
                daily = units_sold / 30.0
                if daily > 0:
                    inv_days = inv_level / daily
            elif inv_level is not None:
                if inv_level > 200:
                    inv_days = 90.0
                elif inv_level > 100:
                    inv_days = 60.0
                elif inv_level > 30:
                    inv_days = 30.0
                else:
                    inv_days = 10.0
        
        out = {}
        for lvl, price in input_candidates.items():
            if inv_days is None:
                out[lvl] = price
            else:
                if inv_days < self.thresholds["low"]:
                    if lvl == "conservative":
                        out[lvl] = round(price * 1.01, 2)
                    elif lvl == "neutral":
                        out[lvl] = round(price * 1.03, 2)
                    else:
                        out[lvl] = round(price * 1.06, 2)
                elif inv_days > self.thresholds["high"]:
                    if lvl == "conservative":
                        out[lvl] = round(price * 0.995, 2)
                    elif lvl == "neutral":
                        out[lvl] = round(price * 0.97, 2)
                    else:
                        out[lvl] = round(price * 0.94, 2)
                else:
                    out[lvl] = price
        
        return AgentTrace("inventory", out, {"inventory_days": inv_days})

# -----------------------------
# Candidate Assembler
# -----------------------------
class CandidateAssembler:
    def __init__(self, guardrails: Dict[str, Any]):
        self.guard = guardrails
    
    def assemble(self, traces: Dict[str, AgentTrace]) -> Dict[str, float]:
        """Combine agent traces into final candidates with distinct pricing strategies"""
        weights = {"base": 0.25, "promo": 0.30, "competitor": 0.25, "inventory": 0.20}
        levels = ["conservative", "neutral", "aggressive"]
        combined = {lvl: 0.0 for lvl in levels}
        
        for lvl in levels:
            s = 0.0
            tw = 0.0
            for aname, w in weights.items():
                price = traces[aname].candidates.get(lvl)
                if price is None:
                    continue
                s += price * w
                tw += w
            combined[lvl] = round(s / tw if tw > 0 else 0.0, 2)
        
        # Ensure distinct prices by applying additional adjustments
        # Conservative: reduce by 3-5%
        # Neutral: as calculated
        # Aggressive: increase by 5-8%
        neutral_price = combined["neutral"]
        conservative_price = max(combined["conservative"], neutral_price * 0.95)  # At least 5% lower
        aggressive_price = max(combined["aggressive"], neutral_price * 1.05)      # At least 5% higher
        
        # Map to frontend-compatible keys
        return {
            "price_base": round(conservative_price, 2),
            "price_optimal": round(neutral_price, 2),
            "price_aggressive": round(aggressive_price, 2)
        }

# -----------------------------
# Units prediction (elasticity fallback)
# -----------------------------
def predict_units_for_candidate(row: pd.Series, price: float, elasticity: float) -> float:
    """Predict units using elasticity formula"""
    hist_units = safe_float(row.get("units_sold"), 0.0)
    hist_price = safe_float(row.get("local_price"), price)
    
    if hist_units <= 0:
        hist_units = max(1.0, safe_float(row.get("revenue"), 0.0) / (hist_price + 1e-9))
    
    units_new = hist_units * ((hist_price + 1e-9) / (price + 1e-9)) ** max(abs(elasticity), 0.1)
    return max(units_new, 0.0)

# -----------------------------
# Selector & SOP enforcement
# -----------------------------
def selector_and_sop(row: pd.Series, candidate_prices: Dict[str, float], guardrails: Dict[str, Any]) -> Dict[str, Any]:
    """Select final price from candidates based on margin requirements"""
    min_margin = guardrails.get("min_margin_pct", 12.0)
    cost = safe_float(row.get("cost"), None)
    if cost is None or math.isnan(cost):
        prev = safe_float(row.get("local_price"), candidate_prices.get("price_base", 100.0))
        cost = prev * (1.0 - min_margin/100.0)
    
    chosen = None
    chosen_key = None
    chosen_margin = None
    priority = ["price_optimal", "price_aggressive", "price_base"]
    
    for key in priority:
        p = candidate_prices.get(key)
        if p is None:
            continue
        margin_pct = (p - cost) / (p + 1e-9) * 100.0
        if margin_pct >= min_margin:
            chosen = p
            chosen_key = key
            chosen_margin = margin_pct
            break
    
    if chosen is None:
        best_m = -math.inf
        best_k = None
        best_p = None
        for k, v in candidate_prices.items():
            if v is None:
                continue
            m = (v - cost) / (v + 1e-9) * 100.0
            if m > best_m:
                best_m = m
                best_k = k
                best_p = v
        if best_p is None:
            best_p = float(row.get("local_price", 100.0))
            best_k = "fallback"
            best_m = (best_p - cost) / (best_p + 1e-9) * 100.0
        chosen = best_p
        chosen_key = best_k
        chosen_margin = best_m
    
    prev_price = float(row.get("local_price", chosen))
    change_pct = pct_change(prev_price, chosen)
    
    if abs(change_pct) <= guardrails.get("auto_approve_pct", 2.0):
        approval = "Auto-approve"
    elif abs(change_pct) <= guardrails.get("manager_review_pct", 5.0):
        approval = "Manager Review"
    else:
        approval = "Committee Review"
    
    return {
        "price_recommended": round(chosen, 2),
        "source_key": chosen_key,
        "margin_pct": round(chosen_margin, 2),
        "change_pct": round(change_pct, 2),
        "approval": approval
    }

# -----------------------------
# Helper: Find or create row
# -----------------------------
def find_or_create_row(df: pd.DataFrame, sku_id: str, channel: str, region: str) -> pd.Series:
    """Find exact match or create synthetic row from aggregated medians"""
    mask_exact = (
        (df["sku_id"].astype(str) == sku_id) &
        (df["channel"].astype(str).str.upper() == channel.upper()) &
        (df["region"].astype(str).str.title() == region.title())
    )
    
    if mask_exact.sum() > 0:
        matches = df[mask_exact]
        if "date" in matches.columns:
            matches = matches.sort_values("date", ascending=False)
        return matches.iloc[0]
    
    # Try partial SKU match
    if sku_id:
        alt = df[df["sku_id"].astype(str).str.contains(sku_id, case=False, na=False)]
        if len(alt) > 0:
            row = alt.iloc[0].copy()
            row["sku_id"] = sku_id
            row["channel"] = channel
            row["region"] = region
            return row
    
    # Create synthetic row from aggregated medians
    row = pd.Series()
    row["sku_id"] = sku_id or "SYNTHETIC_SKU"
    row["channel"] = channel or df["channel"].mode().iloc[0] if len(df) > 0 else "GT"
    row["region"] = region or df["region"].mode().iloc[0] if len(df) > 0 else "North"
    row["category"] = df["category"].mode().iloc[0] if len(df) > 0 and "category" in df.columns else "Dishwash"
    
    for col in ["local_price", "competitor_price", "promo_depth_pct", "final_price", "units_sold",
                "inventory_level", "revenue", "margin", "festival_lift", "seasonality_index", "base_price"]:
        if col in df.columns:
            row[col] = float(df[col].median(skipna=True))
        else:
            row[col] = 100.0 if "price" in col else (1.0 if "lift" in col or "index" in col else 0.0)
    
    if "date" in df.columns:
        row["date"] = df["date"].max() if len(df) > 0 else pd.Timestamp.now()
    
    return row

# -----------------------------
# Advanced Forecasting Engine
# -----------------------------
class ForecastEngine:
    """
    Advanced forecasting engine using ensemble methods:
    - Holt-Winters Exponential Smoothing for seasonality
    - Gradient Boosting for feature-based predictions
    - ARIMA for time series patterns
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.scaler = StandardScaler() if ML_AVAILABLE else None
    
    def prepare_features(self, sku_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for ML-based forecasting"""
        if len(sku_data) < 10:
            return None, None
        
        # Create lag features
        sku_data = sku_data.sort_values('date')
        sku_data['price_lag1'] = sku_data['local_price'].shift(1)
        sku_data['price_lag7'] = sku_data['local_price'].shift(7)
        sku_data['units_lag1'] = sku_data['units_sold'].shift(1)
        sku_data['units_lag7'] = sku_data['units_sold'].shift(7)
        sku_data['revenue_lag1'] = sku_data['revenue'].shift(1)
        
        # Add time-based features
        sku_data['day_of_week'] = sku_data['date'].dt.dayofweek
        sku_data['day_of_month'] = sku_data['date'].dt.day
        sku_data['month'] = sku_data['date'].dt.month
        
        # Drop rows with NaN from lagging
        sku_data = sku_data.dropna()
        
        if len(sku_data) < 5:
            return None, None
        
        feature_cols = ['local_price', 'price_lag1', 'price_lag7', 'units_lag1', 'units_lag7',
                       'revenue_lag1', 'seasonality_index', 'festival_lift', 'promo_depth_pct',
                       'day_of_week', 'day_of_month', 'month']
        
        X = sku_data[feature_cols].values
        y = sku_data['units_sold'].values
        
        return X, y
    
    def forecast_demand(self, sku_id: str, channel: str, region: str, 
                       forecast_horizon: int = 30,
                       price_scenario: Optional[float] = None) -> Dict[str, Any]:
        """
        Forecast demand using ensemble of methods
        
        Args:
            sku_id: SKU identifier
            channel: Sales channel
            region: Geographic region
            forecast_horizon: Number of days to forecast
            price_scenario: Optional price for what-if analysis
        
        Returns:
            Dictionary with forecast results and confidence intervals
        """
        # Filter data for specific SKU-channel-region
        mask = (
            (self.df['sku_id'].astype(str) == sku_id) &
            (self.df['channel'].astype(str) == channel) &
            (self.df['region'].astype(str) == region)
        )
        sku_data = self.df[mask].copy()
        
        if len(sku_data) < 10:
            return {
                "error": "Insufficient historical data for forecasting",
                "min_required": 10,
                "available": len(sku_data)
            }
        
        sku_data = sku_data.sort_values('date')
        
        forecasts = {}
        
        # Method 1: Exponential Smoothing (if enough data)
        if ML_AVAILABLE and len(sku_data) >= 14:
            try:
                ts_data = sku_data.set_index('date')['units_sold']
                model = ExponentialSmoothing(
                    ts_data,
                    seasonal_periods=7,
                    trend='add',
                    seasonal='add',
                    initialization_method="estimated"
                )
                fitted = model.fit()
                hw_forecast = fitted.forecast(steps=forecast_horizon)
                forecasts['exponential_smoothing'] = hw_forecast.values.tolist()
            except Exception as e:
                print(f"Exponential Smoothing failed: {e}")
        
        # Method 2: Gradient Boosting (feature-based)
        if ML_AVAILABLE:
            try:
                X, y = self.prepare_features(sku_data)
                if X is not None and len(X) >= 5:
                    model = GradientBoostingRegressor(
                        n_estimators=100,
                        learning_rate=0.1,
                        max_depth=3,
                        random_state=42
                    )
                    model.fit(X, y)
                    
                    # Generate future predictions
                    last_row = sku_data.iloc[-1]
                    gb_predictions = []
                    
                    for i in range(forecast_horizon):
                        future_features = [
                            price_scenario if price_scenario else last_row['local_price'],
                            last_row['local_price'],
                            sku_data.iloc[-7]['local_price'] if len(sku_data) >= 7 else last_row['local_price'],
                            last_row['units_sold'],
                            sku_data.iloc[-7]['units_sold'] if len(sku_data) >= 7 else last_row['units_sold'],
                            last_row['revenue'],
                            last_row['seasonality_index'],
                            last_row['festival_lift'],
                            last_row['promo_depth_pct'],
                            (last_row['date'] + timedelta(days=i+1)).dayofweek,
                            (last_row['date'] + timedelta(days=i+1)).day,
                            (last_row['date'] + timedelta(days=i+1)).month
                        ]
                        pred = model.predict([future_features])[0]
                        gb_predictions.append(max(0, pred))
                    
                    forecasts['gradient_boosting'] = gb_predictions
            except Exception as e:
                print(f"Gradient Boosting failed: {e}")
        
        # Method 3: Simple moving average with elasticity (fallback)
        try:
            recent_avg = sku_data['units_sold'].tail(7).mean()
            
            # Get category for consistent elasticity estimation
            category = sku_data['category'].mode().iloc[0] if 'category' in sku_data.columns and not sku_data['category'].empty else None
            elasticity = estimate_elasticity(self.df, sku=sku_id, category=category)
            
            if price_scenario:
                current_price = sku_data['local_price'].tail(7).mean()
                price_ratio = current_price / price_scenario
                adjusted_demand = recent_avg * (price_ratio ** abs(elasticity))
            else:
                adjusted_demand = recent_avg
            
            ma_forecast = [adjusted_demand] * forecast_horizon
            forecasts['moving_average'] = ma_forecast
        except Exception as e:
            print(f"Moving average failed: {e}")
        
        # Ensemble: Average available forecasts
        if forecasts:
            ensemble = []
            for i in range(forecast_horizon):
                day_forecasts = [f[i] for f in forecasts.values() if i < len(f)]
                if day_forecasts:
                    ensemble.append(np.mean(day_forecasts))
                else:
                    ensemble.append(recent_avg if 'recent_avg' in locals() else 0)
            
            # Calculate confidence intervals (±20% for simplicity)
            lower_bound = [max(0, x * 0.8) for x in ensemble]
            upper_bound = [x * 1.2 for x in ensemble]
            
            # Generate forecast dates
            last_date = sku_data['date'].max()
            forecast_dates = [(last_date + timedelta(days=i+1)).strftime('%Y-%m-%d') 
                            for i in range(forecast_horizon)]
            
            return {
                "sku_id": sku_id,
                "channel": channel,
                "region": region,
                "forecast_horizon_days": forecast_horizon,
                "forecast_dates": forecast_dates,
                "forecasted_demand": ensemble,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "methods_used": list(forecasts.keys()),
                "historical_avg": float(sku_data['units_sold'].mean()),
                "recent_trend": float(sku_data['units_sold'].tail(7).mean()),
                "price_scenario": price_scenario,
                "elasticity_used": elasticity if 'elasticity' in locals() else -0.8
            }
        else:
            return {"error": "All forecasting methods failed"}

# -----------------------------
# Pydantic Models
# -----------------------------
class SingleSKURequest(BaseModel):
    sku_id: str
    channel: str
    region: str

class WhatIfRequest(BaseModel):
    sku_id: str
    channel: str
    region: str
    nl_query: Optional[str] = Field(None, description="Natural language query for what-if scenario")
    # Price scenarios
    price_change_pct: Optional[float] = Field(None, description="Price change percentage (e.g., 5 for 5% increase)")
    target_price: Optional[float] = Field(None, description="Specific target price")
    # Market conditions
    promo_depth_pct: Optional[float] = Field(None, description="Promotional depth percentage")
    competitor_price: Optional[float] = Field(None, description="Competitor price scenario")
    competitor_price_change_pct: Optional[float] = Field(None, description="Competitor price change %")
    # Operational factors
    inv_days: Optional[float] = Field(None, description="Inventory days scenario")
    inventory_change_pct: Optional[float] = Field(None, description="Inventory level change %")
    # Seasonal factors
    seasonality_index: Optional[float] = Field(None, description="Seasonality multiplier")
    festival_lift: Optional[float] = Field(None, description="Festival lift multiplier")
    # Market elasticity
    demand_elasticity: Optional[float] = Field(None, description="Custom demand elasticity")
    # Forecast integration
    include_forecast: Optional[bool] = Field(False, description="Include demand forecast")
    forecast_days: Optional[int] = Field(30, description="Forecast horizon in days")
    # SOP overrides
    sop_overrides: Optional[Dict[str, float]] = None
    # Advanced overrides
    base_uplift_multiplicative: Optional[float] = Field(None, description="Multiplicative uplift for base price (e.g. 1.05 for 5% increase)")

class ForecastRequest(BaseModel):
    sku_id: str
    channel: str
    region: str
    forecast_horizon: int = Field(30, description="Number of days to forecast", ge=7, le=90)
    price_scenario: Optional[float] = Field(None, description="Price for what-if forecast")

class ForecastPricingRequest(BaseModel):
    sku_id: str
    channel: str
    region: str
    start_date: str
    days: int = 30

class GuardrailsUpdate(BaseModel):
    min_margin_pct: Optional[float] = None
    max_promo_depth_pct: Optional[float] = None
    auto_approve_pct: Optional[float] = None
    manager_review_pct: Optional[float] = None
    max_monthly_change_pct: Optional[float] = None

# -----------------------------
# Dataset loader
# -----------------------------
def load_dataset(path: str = DATA_CSV) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"Warning: Dataset not found at {path}. Creating sample dataset.")
        sample_data = {
            "date": pd.date_range("2024-01-01", periods=16, freq="D"),
            "sku_id": ["SKU_0001"] * 16,
            "category": ["Dishwash"] * 16,
            "channel": ["GT"] * 4 + ["MT"] * 4 + ["ECOM"] * 4 + ["QCOM"] * 4,
            "region": ["North", "South", "East", "West"] * 4,
            "local_price": [114.82, 145.26, 150.73, 109.78] * 4,
            "base_price": [128.1] * 16,
            "competitor_price": [127.95, 137.32, 135.55, 131.08] * 4,
            "promo_flag": [0] * 16,
            "promo_depth_pct": [0.0] * 16,
            "final_price": [114.82, 145.26, 150.73, 109.78] * 4,
            "units_sold": [2489, 2056, 2843, 1299] * 4,
            "inventory_level": [1200, 3000, 600, 300] * 4,
            "is_stockout": [0] * 16,
            "revenue": [285786.98, 298718.76, 428414.72, 142654.75] * 4,
            "margin": [92715.25, 116832.94, 113546.12, 116055.06] * 4,
            "festival_lift": [1.3] * 16,
            "seasonality_index": [1.0] * 16
        }
        df = pd.DataFrame(sample_data)
        data_dir = os.path.dirname(path) if os.path.dirname(path) else "."
        if data_dir and not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)
        df.to_csv(path, index=False)
        print(f"Created sample dataset with {len(df)} rows at {path}")
        return df
    
    # Try reading with multiple encodings
    df = None
    for encoding in ['utf-8', 'cp1252', 'latin1', 'iso-8859-1']:
        try:
            df = pd.read_csv(path, encoding=encoding)
            print(f"Successfully loaded dataset with {encoding} encoding")
            break
        except UnicodeDecodeError:
            continue
            
    if df is None:
        raise ValueError(f"Could not decode dataset at {path}. Please check encoding.")
    # Ensure date column is datetime
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    
    # Enrich seasonality
    df = enrich_seasonality(df)
    
    print(f"Loaded dataset with {len(df)} rows from {path}")
    return df

# Initialize dataset on startup
@app.on_event("startup")
async def startup_event():
    global dataset_df
    data_dir = os.path.dirname(DATA_CSV) if os.path.dirname(DATA_CSV) else "."
    if data_dir and not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
    dataset_df = load_dataset()

# -----------------------------
# API Endpoints
# -----------------------------
@app.get("/")
async def root():
    return {"message": "Pricing Simulator API", "version": "2.0.0", "type": "hybrid"}

@app.get("/health")
async def health():
    return {"status": "healthy", "dataset_loaded": dataset_df is not None}

@app.get("/dataset/info")
async def dataset_info():
    if dataset_df is None:
        raise HTTPException(status_code=500, detail="Dataset not loaded")
    return {
        "rows": len(dataset_df),
        "columns": list(dataset_df.columns),
        "skus": sorted(dataset_df["sku_id"].unique().tolist()) if "sku_id" in dataset_df.columns else [],
        "channels": sorted(dataset_df["channel"].unique().tolist()) if "channel" in dataset_df.columns else [],
        "regions": sorted(dataset_df["region"].unique().tolist()) if "region" in dataset_df.columns else []
    }

@app.get("/dataset/skus")
async def get_skus():
    if dataset_df is None:
        raise HTTPException(status_code=500, detail="Dataset not loaded")
    skus = sorted(dataset_df["sku_id"].unique().tolist()) if "sku_id" in dataset_df.columns else []
    return {"skus": skus}

@app.get("/dataset/channels")
async def get_channels():
    if dataset_df is None:
        raise HTTPException(status_code=500, detail="Dataset not loaded")
    channels = sorted(dataset_df["channel"].unique().tolist()) if "channel" in dataset_df.columns else []
    return {"channels": channels}

@app.get("/dataset/regions")
async def get_regions():
    if dataset_df is None:
        raise HTTPException(status_code=500, detail="Dataset not loaded")
    regions = sorted(dataset_df["region"].unique().tolist()) if "region" in dataset_df.columns else []
    return {"regions": regions}

@app.post("/pricing/single-sku")
async def single_sku_pricing(request: SingleSKURequest):
    """Single SKU pricing analysis with hybrid agents"""
    if dataset_df is None:
        raise HTTPException(status_code=500, detail="Dataset not loaded")
    
    row = find_or_create_row(dataset_df, request.sku_id, request.channel, request.region)
    
    # Run agents
    base_tr = BasePriceAgent().run(row, dataset_df)
    promo_tr = PromoAgent(current_guardrails).run(row, base_tr.candidates)
    comp_tr = CompetitorAgent().run(row, promo_tr.candidates)
    inv_tr = InventoryAgent().run(row, comp_tr.candidates)
    
    traces = {"base": base_tr, "promo": promo_tr, "competitor": comp_tr, "inventory": inv_tr}
    
    # Assemble candidates
    assembler = CandidateAssembler(current_guardrails)
    mapped = assembler.assemble(traces)
    
    # Estimate elasticity
    elasticity = estimate_elasticity(dataset_df, sku=row.get("sku_id"), category=row.get("category"))
    
    # Predict units for each candidate
    candidates_detail = {}
    for k, price in mapped.items():
        pred_units = predict_units_for_candidate(row, price, elasticity)
        candidates_detail[k] = {"price": round(price, 2), "pred_units": pred_units, "elasticity": elasticity}
    
    # Select final price
    selection = selector_and_sop(row, mapped, current_guardrails)
    
    # Format for frontend compatibility
    # Map agent outputs to show progression through the pipeline
    base_price = base_tr.candidates.get("neutral", mapped["price_base"])
    promo_price = promo_tr.candidates.get("neutral", mapped["price_optimal"])
    comp_price = comp_tr.candidates.get("neutral", mapped["price_optimal"])
    inv_price = inv_tr.candidates.get("neutral", mapped["price_optimal"])
    
    return {
        "sku_id": request.sku_id,
        "channel": request.channel,
        "region": request.region,
        "agent_outputs": {
            "base": {
                "price": base_price,
                "candidates": base_tr.candidates,
                "reason": f"base from prev={base_tr.details.get('prev_median', 0):.2f}, seasonality={base_tr.details.get('seasonality', 1.0):.3f}, festival={base_tr.details.get('festival', 1.0):.3f}",
                "meta": base_tr.details
            },
            "promo": {
                "price": promo_price,
                "candidates": promo_tr.candidates,
                "reason": f"Promo optimization applied",
                "meta": promo_tr.details
            },
            "comp": {
                "price": comp_price,
                "candidates": comp_tr.candidates,
                "reason": f"Competitive positioning",
                "meta": comp_tr.details
            },
            "inventory": {
                "price": inv_price,
                "candidates": inv_tr.candidates,
                "reason": f"Inventory adjustment",
                "meta": inv_tr.details
            }
        },
        "candidates": {
            # Return the three different price strategies
            "price_base": mapped["price_base"],           # Conservative (from assembler)
            "price_optimal": mapped["price_optimal"],     # Neutral/Optimal (from assembler)
            "price_aggressive": mapped["price_aggressive"]  # Aggressive (from assembler)
        },
        "selection": selection,
        "reasons": {
            "base": f"base from prev={base_tr.details.get('prev_median', 0):.2f}, seasonality={base_tr.details.get('seasonality', 1.0):.3f}, festival={base_tr.details.get('festival', 1.0):.3f}",
            "promo": promo_tr.details,
            "comp": comp_tr.details,
            "inv": inv_tr.details
        },
        "elasticity": elasticity,
        "candidates_detail": candidates_detail
    }

@app.post("/pricing/what-if")
async def what_if_pricing(request: WhatIfRequest):
    """What-If scenario analysis"""
    if dataset_df is None:
        raise HTTPException(status_code=500, detail="Dataset not loaded")
    
    row = find_or_create_row(dataset_df, request.sku_id, request.channel, request.region)
    
    # Original run
    def run_pipeline(r):
        base_tr = BasePriceAgent().run(r, dataset_df)
        promo_tr = PromoAgent(current_guardrails).run(r, base_tr.candidates)
        comp_tr = CompetitorAgent().run(r, promo_tr.candidates)
        inv_tr = InventoryAgent().run(r, comp_tr.candidates)
        traces = {"base": base_tr, "promo": promo_tr, "competitor": comp_tr, "inventory": inv_tr}
        mapped = CandidateAssembler(current_guardrails).assemble(traces)
        elasticity = estimate_elasticity(dataset_df, sku=r.get("sku_id"), category=r.get("category"))
        details = {}
        for k, pr in mapped.items():
            pred_units = predict_units_for_candidate(r, pr, elasticity)
            details[k] = {"price": round(pr, 2), "pred_units": pred_units, "elasticity": elasticity}
        sel = selector_and_sop(r, mapped, current_guardrails)
        return {
            "traces": traces,
            "candidates": mapped,
            "candidates_detail": details,
            "selection": sel
        }
    
    orig = run_pipeline(row)
    
    # Apply overrides
    wrow = row.copy()
    overrides = {}
    if request.promo_depth_pct is not None:
        wrow["promo_depth_pct"] = request.promo_depth_pct
        overrides["promo_depth_pct"] = request.promo_depth_pct
    if request.competitor_price is not None:
        wrow["competitor_price"] = request.competitor_price
        overrides["competitor_price"] = request.competitor_price
    if request.inv_days is not None:
        wrow["inventory_days"] = request.inv_days
        overrides["inv_days"] = request.inv_days
    if request.seasonality_index is not None:
        wrow["seasonality_index"] = request.seasonality_index
        overrides["seasonality_index"] = request.seasonality_index
    if request.festival_lift is not None:
        wrow["festival_lift"] = request.festival_lift
        overrides["festival_lift"] = request.festival_lift
    
    local_guard = current_guardrails.copy()
    if request.sop_overrides:
        local_guard.update(request.sop_overrides)
    
    what = run_pipeline(wrow)
    
    # Calculate delta
    delta = {
        "price_change_pct": pct_change(
            orig["selection"]["price_recommended"],
            what["selection"]["price_recommended"]
        ),
        "margin_change_pct": (what["selection"]["margin_pct"] - orig["selection"]["margin_pct"])
        if (orig["selection"].get("margin_pct") is not None) else None
    }
    
    # Map candidates for frontend compatibility
    orig_base_price = orig["traces"]["base"].candidates.get("neutral", orig["candidates"]["price_base"])
    orig_promo_price = orig["traces"]["promo"].candidates.get("neutral", orig["candidates"]["price_optimal"])
    orig_comp_price = orig["traces"]["competitor"].candidates.get("neutral", orig["candidates"]["price_optimal"])
    orig_inv_price = orig["traces"]["inventory"].candidates.get("neutral", orig["candidates"]["price_optimal"])
    
    what_base_price = what["traces"]["base"].candidates.get("neutral", what["candidates"]["price_base"])
    what_promo_price = what["traces"]["promo"].candidates.get("neutral", what["candidates"]["price_optimal"])
    what_comp_price = what["traces"]["competitor"].candidates.get("neutral", what["candidates"]["price_optimal"])
    what_inv_price = what["traces"]["inventory"].candidates.get("neutral", what["candidates"]["price_optimal"])
    
    return {
        "original": {
            "candidates": {
                "price_base": orig["candidates"]["price_base"],
                "price_promo": orig_promo_price,
                "price_comp": orig_comp_price,
                "price_inventory": orig_inv_price
            },
            "selection": orig["selection"],
            "reasons": {
                "base": f"base from prev={orig['traces']['base'].details.get('prev_median', 0):.2f}",
                "promo": orig["traces"]["promo"].details,
                "comp": orig["traces"]["competitor"].details,
                "inv": orig["traces"]["inventory"].details
            }
        },
        "what_if": {
            "overrides": overrides,
            "candidates": {
                "price_base": what["candidates"]["price_base"],
                "price_promo": what_promo_price,
                "price_comp": what_comp_price,
                "price_inventory": what_inv_price
            },
            "selection": what["selection"],
            "reasons": {
                "base": f"base from prev={what['traces']['base'].details.get('prev_median', 0):.2f}",
                "promo": what["traces"]["promo"].details,
                "comp": what["traces"]["competitor"].details,
                "inv": what["traces"]["inventory"].details
            }
        },
        "delta": delta
    }

@app.post("/pricing/process_csv")
async def process_csv(file: UploadFile = File(...)):
    """Process uploaded CSV and return results with pricing recommendations"""
    try:
        contents = await file.read()
        
        # Robust encoding detection
        s = None
        for encoding in ['utf-8', 'cp1252', 'latin1', 'iso-8859-1']:
            try:
                s = str(contents, encoding)
                print(f"Successfully decoded CSV using {encoding}")
                break
            except UnicodeDecodeError:
                continue
                
        if s is None:
            return JSONResponse(status_code=400, content={"error": "Could not decode file. Please check encoding."})
            
        data = io.StringIO(s)
        df = pd.read_csv(data)
        
        # Ensure required columns exist
        required_cols = ["sku_id", "channel", "region"]
        for col in required_cols:
            if col not in df.columns:
                return JSONResponse(status_code=400, content={"error": f"Missing required column: {col}"})
        
        rows_out = []
        
        for idx, row in df.iterrows():
            try:
                # Ensure minimal fields for agents
                if "seasonality_index" not in row:
                    row["seasonality_index"] = 1.0
                if "festival_lift" not in row:
                    row["festival_lift"] = 1.0
                if "local_price" not in row:
                    row["local_price"] = 100.0 # Default
                
                base_tr = BasePriceAgent().run(row, dataset_df)
                promo_tr = PromoAgent(current_guardrails).run(row, base_tr.candidates)
                comp_tr = CompetitorAgent().run(row, promo_tr.candidates)
                inv_tr = InventoryAgent().run(row, comp_tr.candidates)
                
                traces = {"base": base_tr, "promo": promo_tr, "competitor": comp_tr, "inventory": inv_tr}
                mapped = CandidateAssembler(current_guardrails).assemble(traces)
                selection = selector_and_sop(row, mapped, current_guardrails)
                
                # Construct output row
                out_row = row.to_dict()
                out_row.update({
                    "recommended_price_base": mapped["price_base"],
                    "recommended_price_optimal": mapped["price_optimal"],
                    "recommended_price_aggressive": mapped["price_aggressive"],
                    "final_recommended_price": selection["price_recommended"],
                    "approval_status": selection["approval"],
                    "projected_margin_pct": selection["margin_pct"],
                    "price_change_pct": selection["change_pct"]
                })
                rows_out.append(out_row)
                
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                continue
        
        out_df = pd.DataFrame(rows_out)
        
        # Save to output dir
        ensure_output_dir()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"processed_{timestamp}_{file.filename}"
        out_path = os.path.join(OUTPUT_DIR, filename)
        out_df.to_csv(out_path, index=False)
        
        return FileResponse(out_path, media_type='text/csv', filename=filename)
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/pricing/batch")
async def batch_pricing(
    limit: Optional[int] = Query(None, description="Max rows to process"),
    sku_id: Optional[str] = Query(None, description="Filter by SKU ID"),
    channel: Optional[str] = Query(None, description="Filter by Channel"),
    region: Optional[str] = Query(None, description="Filter by Region")
):
    """Batch pricing analysis with optional filters and deduplication"""
    if dataset_df is None:
        raise HTTPException(status_code=500, detail="Dataset not loaded")
    
    # Apply filters
    subset = dataset_df.copy()
    if sku_id:
        subset = subset[subset["sku_id"].astype(str) == sku_id]
    if channel:
        subset = subset[subset["channel"].astype(str) == channel]
    if region:
        subset = subset[subset["region"].astype(str) == region]
    
    # Deduplicate: Keep only the most recent row for each SKU-Channel-Region combination
    # Sort by date descending to get most recent first
    if "date" in subset.columns:
        subset = subset.sort_values("date", ascending=False)
    
    # Group by SKU-Channel-Region and take the first (most recent) row
    subset = subset.groupby(["sku_id", "channel", "region"], as_index=False).first()
    
    # Apply limit after deduplication
    if limit:
        subset = subset.head(limit)
    else:
        subset = subset.head(120)  # Default to 120 unique SKU-Channel-Region combinations
    
    rows_out = []
    seen_combinations = set()  # Track unique combinations
    
    for idx, row in subset.iterrows():
        try:
            # Create unique key
            combo_key = (str(row.get("sku_id", "")), str(row.get("channel", "")), str(row.get("region", "")))
            
            # Skip if we've already processed this combination
            if combo_key in seen_combinations:
                continue
            seen_combinations.add(combo_key)
            
            base_tr = BasePriceAgent().run(row, dataset_df)
            promo_tr = PromoAgent(current_guardrails).run(row, base_tr.candidates)
            comp_tr = CompetitorAgent().run(row, promo_tr.candidates)
            inv_tr = InventoryAgent().run(row, comp_tr.candidates)
            traces = {"base": base_tr, "promo": promo_tr, "competitor": comp_tr, "inventory": inv_tr}
            mapped = CandidateAssembler(current_guardrails).assemble(traces)
            selection = selector_and_sop(row, mapped, current_guardrails)
            
            rows_out.append({
                "sku_id": str(row.get("sku_id", "")),
                "region": str(row.get("region", "")),
                "channel": str(row.get("channel", "")),
                "date": str(row.get("date", "")) if "date" in row.index else None,
                "price_base": mapped["price_base"],
                "price_promo": mapped["price_optimal"],
                "price_comp": mapped["price_optimal"],
                "price_inventory": mapped["price_optimal"],
                "final_price": selection["price_recommended"],
                "approval": selection["approval"],
                "margin_pct": selection["margin_pct"],
                "change_pct": selection["change_pct"]
            })
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            continue
    
    return {
        "count": len(rows_out),
        "results": rows_out,
        "note": "Showing most recent recommendation per unique SKU-Channel-Region combination"
    }

@app.post("/pricing/forecast")
async def forecast_demand(request: ForecastRequest):
    """
    Advanced demand forecasting using ensemble methods
    - Exponential Smoothing for seasonality
    - Gradient Boosting for feature-based predictions
    - Moving Average with elasticity as fallback
    """
    if dataset_df is None:
        raise HTTPException(status_code=500, detail="Dataset not loaded")
    
    if not ML_AVAILABLE:
        return {
            "error": "ML libraries not available",
            "message": "Install scikit-learn and statsmodels: pip install scikit-learn statsmodels"
        }
    
    engine = ForecastEngine(dataset_df)
    result = engine.forecast_demand(
        sku_id=request.sku_id,
        channel=request.channel,
        region=request.region,
        forecast_horizon=request.forecast_horizon,
        price_scenario=request.price_scenario
    )
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result

@app.post("/pricing/what-if-enhanced")
async def what_if_pricing_enhanced(request: WhatIfRequest):
    """
    Enhanced What-If scenario analysis with realistic modeling:
    - Multiple price scenarios (percentage or absolute)
    - Competitor response modeling
    - Inventory impact analysis
    - Demand elasticity integration
    - Optional demand forecasting
    """
    if dataset_df is None:
        raise HTTPException(status_code=500, detail="Dataset not loaded")
    
    row = find_or_create_row(dataset_df, request.sku_id, request.channel, request.region)
    
    # Enhanced pipeline with scenario modeling
    def run_enhanced_pipeline(r, custom_elasticity=None, base_uplift=None):
        base_tr = BasePriceAgent().run(r, dataset_df)
        
        # Apply base uplift if provided
        if base_uplift is not None:
            for k in base_tr.candidates:
                base_tr.candidates[k] *= base_uplift
            base_tr.details["base_uplift_applied"] = base_uplift
            
        promo_tr = PromoAgent(current_guardrails).run(r, base_tr.candidates)
        comp_tr = CompetitorAgent().run(r, promo_tr.candidates)
        inv_tr = InventoryAgent().run(r, comp_tr.candidates)
        traces = {"base": base_tr, "promo": promo_tr, "competitor": comp_tr, "inventory": inv_tr}
        mapped = CandidateAssembler(current_guardrails).assemble(traces)
        
        # Use custom or estimated elasticity
        elasticity = custom_elasticity if custom_elasticity else estimate_elasticity(
            dataset_df, sku=r.get("sku_id"), category=r.get("category")
        )
        
        details = {}
        for k, pr in mapped.items():
            pred_units = predict_units_for_candidate(r, pr, elasticity)
            pred_revenue = pr * pred_units
            cost = safe_float(r.get("cost"), pr * 0.7)  # Estimate cost if missing
            pred_margin = pred_revenue - (cost * pred_units)
            pred_margin_pct = (pred_margin / pred_revenue * 100) if pred_revenue > 0 else 0
            
            details[k] = {
                "price": round(pr, 2),
                "pred_units": round(pred_units, 2),
                "pred_revenue": round(pred_revenue, 2),
                "pred_margin": round(pred_margin, 2),
                "pred_margin_pct": round(pred_margin_pct, 2),
                "elasticity": round(elasticity, 3)
            }
        
        sel = selector_and_sop(r, mapped, current_guardrails)
        
        return {
            "traces": traces,
            "candidates": mapped,
            "candidates_detail": details,
            "selection": sel,
            "elasticity": elasticity
        }
    
    # Original scenario
    orig = run_enhanced_pipeline(row, request.demand_elasticity)
    
    # Apply what-if scenarios
    wrow = row.copy()
    overrides = {}
    scenario_description = []
    
    # NL Query Processing
    nl_result = {}
    if request.nl_query:
        nl_result = nl_interpret_with_bedrock(request.nl_query, {"sku": request.sku_id, "channel": request.channel, "region": request.region})
        nl_overrides = nl_result.get("overrides", {})
        
        # Apply NL overrides
        if "local_price" in nl_overrides:
            wrow["local_price"] = float(nl_overrides["local_price"])
            overrides["local_price"] = wrow["local_price"]
            scenario_description.append(f"Price set to {wrow['local_price']}")
            
        if "competitor_price" in nl_overrides:
            wrow["competitor_price"] = float(nl_overrides["competitor_price"])
            overrides["competitor_price"] = wrow["competitor_price"]
            scenario_description.append(f"Competitor set to {wrow['competitor_price']}")
            
        if "competitor_pct_change" in nl_overrides:
            base_comp = safe_float(row.get("competitor_price"), 100.0)
            pct = float(nl_overrides["competitor_pct_change"])
            wrow["competitor_price"] = round(base_comp * (1 + pct/100.0), 2)
            overrides["competitor_price"] = wrow["competitor_price"]
            scenario_description.append(f"Competitor changed by {pct}%")

        if "promo_depth_pct" in nl_overrides:
            wrow["promo_depth_pct"] = float(nl_overrides["promo_depth_pct"])
            overrides["promo_depth_pct"] = wrow["promo_depth_pct"]
            scenario_description.append(f"Promo set to {wrow['promo_depth_pct']}%")
            
        if "inventory_days" in nl_overrides:
            wrow["inventory_days"] = float(nl_overrides["inventory_days"])
            overrides["inventory_days"] = wrow["inventory_days"]
            scenario_description.append(f"Inventory days set to {wrow['inventory_days']}")
            
        if "channel" in nl_overrides:
            wrow["channel"] = nl_overrides["channel"]
            overrides["channel"] = wrow["channel"]
            scenario_description.append(f"Channel set to {wrow['channel']}")

    # Price scenarios
    if request.price_change_pct is not None:
        current_price = safe_float(row.get("local_price"), 100.0)
        new_price = current_price * (1 + request.price_change_pct / 100)
        wrow["local_price"] = new_price
        overrides["price_change_pct"] = request.price_change_pct
        overrides["new_price"] = round(new_price, 2)
        scenario_description.append(f"Price {'+' if request.price_change_pct > 0 else ''}{request.price_change_pct}%")
    
    if request.target_price is not None:
        wrow["local_price"] = request.target_price
        overrides["target_price"] = request.target_price
        current_price = safe_float(row.get("local_price"), 100.0)
        change_pct = ((request.target_price - current_price) / current_price) * 100
        overrides["implied_change_pct"] = round(change_pct, 2)
        scenario_description.append(f"Target price ₹{request.target_price}")
    
    # Promotional scenarios
    if request.promo_depth_pct is not None:
        wrow["promo_depth_pct"] = request.promo_depth_pct
        overrides["promo_depth_pct"] = request.promo_depth_pct
        scenario_description.append(f"Promo {request.promo_depth_pct}%")
    
    # Competitor scenarios
    if request.competitor_price is not None:
        wrow["competitor_price"] = request.competitor_price
        overrides["competitor_price"] = request.competitor_price
        scenario_description.append(f"Competitor ₹{request.competitor_price}")
    
    if request.competitor_price_change_pct is not None:
        current_comp = safe_float(row.get("competitor_price"), 100.0)
        new_comp = current_comp * (1 + request.competitor_price_change_pct / 100)
        wrow["competitor_price"] = new_comp
        overrides["competitor_price_change_pct"] = request.competitor_price_change_pct
        overrides["new_competitor_price"] = round(new_comp, 2)
        scenario_description.append(f"Competitor {'+' if request.competitor_price_change_pct > 0 else ''}{request.competitor_price_change_pct}%")
    
    # Inventory scenarios
    if request.inv_days is not None:
        wrow["inventory_days"] = request.inv_days
        overrides["inv_days"] = request.inv_days
        scenario_description.append(f"Inventory {request.inv_days} days")
    
    if request.inventory_change_pct is not None:
        current_inv = safe_float(row.get("inventory_level"), 1000.0)
        new_inv = current_inv * (1 + request.inventory_change_pct / 100)
        wrow["inventory_level"] = new_inv
        overrides["inventory_change_pct"] = request.inventory_change_pct
        overrides["new_inventory_level"] = round(new_inv, 2)
        scenario_description.append(f"Inventory {'+' if request.inventory_change_pct > 0 else ''}{request.inventory_change_pct}%")
    
    # Seasonal scenarios
    if request.seasonality_index is not None:
        wrow["seasonality_index"] = request.seasonality_index
        overrides["seasonality_index"] = request.seasonality_index
        scenario_description.append(f"Seasonality {request.seasonality_index}x")
    
    if request.festival_lift is not None:
        wrow["festival_lift"] = request.festival_lift
        overrides["festival_lift"] = request.festival_lift
        scenario_description.append(f"Festival lift {request.festival_lift}x")
        
    # Base Uplift
    if request.base_uplift_multiplicative is not None:
        overrides["base_uplift_multiplicative"] = request.base_uplift_multiplicative
        scenario_description.append(f"Base Uplift {request.base_uplift_multiplicative}x")
    
    # SOP overrides
    local_guard = current_guardrails.copy()
    if request.sop_overrides:
        local_guard.update(request.sop_overrides)
        overrides["sop_overrides"] = request.sop_overrides
    
    # Run what-if scenario
    what = run_enhanced_pipeline(wrow, request.demand_elasticity, request.base_uplift_multiplicative)
    
    # Calculate comprehensive deltas
    delta = {
        "price_change_pct": pct_change(
            orig["selection"]["price_recommended"],
            what["selection"]["price_recommended"]
        ),
        "margin_change_pct": (what["selection"]["margin_pct"] - orig["selection"]["margin_pct"])
        if (orig["selection"].get("margin_pct") is not None) else None,
        "revenue_impact": None,
        "units_impact": None,
        "margin_impact": None
    }
    
    # Calculate revenue and units impact
    if "price_optimal" in orig["candidates_detail"] and "price_optimal" in what["candidates_detail"]:
        orig_rev = orig["candidates_detail"]["price_optimal"]["pred_revenue"]
        what_rev = what["candidates_detail"]["price_optimal"]["pred_revenue"]
        delta["revenue_impact"] = round(what_rev - orig_rev, 2)
        delta["revenue_change_pct"] = round(pct_change(orig_rev, what_rev), 2)
        
        orig_units = orig["candidates_detail"]["price_optimal"]["pred_units"]
        what_units = what["candidates_detail"]["price_optimal"]["pred_units"]
        delta["units_impact"] = round(what_units - orig_units, 2)
        delta["units_change_pct"] = round(pct_change(orig_units, what_units), 2)
        
        orig_margin = orig["candidates_detail"]["price_optimal"]["pred_margin"]
        what_margin = what["candidates_detail"]["price_optimal"]["pred_margin"]
        delta["margin_impact"] = round(what_margin - orig_margin, 2)
        delta["margin_change_pct"] = round(pct_change(orig_margin, what_margin), 2)
    
    # Optional: Include demand forecast
    forecast_result = None
    if request.include_forecast and ML_AVAILABLE:
        try:
            engine = ForecastEngine(dataset_df)
            forecast_price = overrides.get("new_price") or overrides.get("target_price")
            forecast_result = engine.forecast_demand(
                sku_id=request.sku_id,
                channel=request.channel,
                region=request.region,
                forecast_horizon=request.forecast_days,
                price_scenario=forecast_price
            )
        except Exception as e:
            forecast_result = {"error": str(e)}
    
    # Map candidates for frontend compatibility
    orig_base_price = orig["traces"]["base"].candidates.get("neutral", orig["candidates"]["price_base"])
    orig_promo_price = orig["traces"]["promo"].candidates.get("neutral", orig["candidates"]["price_optimal"])
    orig_comp_price = orig["traces"]["competitor"].candidates.get("neutral", orig["candidates"]["price_optimal"])
    orig_inv_price = orig["traces"]["inventory"].candidates.get("neutral", orig["candidates"]["price_optimal"])
    
    what_base_price = what["traces"]["base"].candidates.get("neutral", what["candidates"]["price_base"])
    what_promo_price = what["traces"]["promo"].candidates.get("neutral", what["candidates"]["price_optimal"])
    what_comp_price = what["traces"]["competitor"].candidates.get("neutral", what["candidates"]["price_optimal"])
    what_inv_price = what["traces"]["inventory"].candidates.get("neutral", what["candidates"]["price_optimal"])
    
    # Generate LLM Explanation
    explanation = "Analysis complete."
    if boto3:
        try:
            prompt = f"""
            Analyze this pricing scenario comparison:
            
            Context: SKU {request.sku_id}, Channel {request.channel}, Region {request.region}
            Scenario: {" | ".join(scenario_description) if scenario_description else "No specific overrides"}
            
            Original:
            - Price: {orig['selection']['price_recommended']}
            - Margin: {orig['selection']['margin_pct']}%
            - Units: {orig['candidates_detail']['price_optimal']['pred_units']:.0f}
            
            What-If:
            - Price: {what['selection']['price_recommended']}
            - Margin: {what['selection']['margin_pct']}%
            - Units: {what['candidates_detail']['price_optimal']['pred_units']:.0f}
            
            Delta:
            - Price Change: {delta.get('price_change_pct', 0):.1f}%
            - Margin Impact: {delta.get('margin_impact', 0):.0f}
            - Revenue Impact: {delta.get('revenue_impact', 0):.0f}
            
            Provide a concise, 2-sentence business explanation of the outcome. Focus on why the price changed (or didn't) and the impact on margin/revenue.
            """
            
            messages = [{"role": "user", "content": prompt}]
            explanation = bedrock_invoke(messages, max_tokens=150)
        except Exception as e:
            explanation = f"Could not generate explanation: {e}"

    return {
        "scenario_description": " | ".join(scenario_description) if scenario_description else "No changes",
        "explanation": explanation,
        "original": {
            "candidates": {
                "price_base": orig["candidates"]["price_base"],
                "price_promo": orig_promo_price,
                "price_comp": orig_comp_price,
                "price_inventory": orig_inv_price
            },
            "candidates_detail": orig["candidates_detail"],
            "selection": orig["selection"],
            "reasons": {
                "base": f"base from prev={orig['traces']['base'].details.get('prev_median', 0):.2f}",
                "promo": orig["traces"]["promo"].details,
                "comp": orig["traces"]["competitor"].details,
                "inv": orig["traces"]["inventory"].details
            },
            "elasticity": round(orig["elasticity"], 3)
        },
        "what_if": {
            "overrides": overrides,
            "candidates": {
                "price_base": what["candidates"]["price_base"],
                "price_promo": what_promo_price,
                "price_comp": what_comp_price,
                "price_inventory": what_inv_price
            },
            "candidates_detail": what["candidates_detail"],
            "selection": what["selection"],
            "reasons": {
                "base": f"base from prev={what['traces']['base'].details.get('prev_median', 0):.2f}",
                "promo": what["traces"]["promo"].details,
                "comp": what["traces"]["competitor"].details,
                "inv": what["traces"]["inventory"].details
            },
            "elasticity": round(what["elasticity"], 3)
        },
        "delta": delta,
        "forecast": forecast_result if request.include_forecast else None
    }

@app.get("/guardrails")
async def get_guardrails():
    """Get current SOP guardrails"""
    return current_guardrails

@app.put("/guardrails")
async def update_guardrails(guardrails: GuardrailsUpdate):
    """Update SOP guardrails"""
    global current_guardrails
    if guardrails.min_margin_pct is not None:
        current_guardrails["min_margin_pct"] = guardrails.min_margin_pct
    if guardrails.max_promo_depth_pct is not None:
        current_guardrails["max_promo_depth_pct"] = guardrails.max_promo_depth_pct
    if guardrails.auto_approve_pct is not None:
        current_guardrails["auto_approve_pct"] = guardrails.auto_approve_pct
    if guardrails.manager_review_pct is not None:
        current_guardrails["manager_review_pct"] = guardrails.manager_review_pct
    if guardrails.max_monthly_change_pct is not None:
        current_guardrails["max_monthly_change_pct"] = guardrails.max_monthly_change_pct
    
    return {"message": "Guardrails updated", "guardrails": current_guardrails}

@app.post("/pricing/forecast-pricing")
async def forecast_pricing_endpoint(request: ForecastPricingRequest):
    """
    Forecasts pricing for a future period (start_date + days).
    Returns 3 candidate prices (conservative, neutral, aggressive) for each day.
    """
    if dataset_df is None:
        raise HTTPException(status_code=500, detail="Dataset not loaded")
    
    df_all = dataset_df
    sku = request.sku_id
    channel = request.channel
    region = request.region
    start_date_str = request.start_date
    days = request.days
    
    try:
        start = pd.to_datetime(start_date_str)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid start_date format. Use YYYY-MM-DD")
    
    # Determine baseline series
    sku_mask = df_all["sku_id"].astype(str) == str(sku)
    if sku_mask.sum() >= 5:
        hist = df_all[sku_mask].copy()
    else:
        # Fallback to category
        cat_matches = df_all[df_all["sku_id"].astype(str)==str(sku)]
        cat_mode = cat_matches["category"].mode().iloc[0] if not cat_matches.empty else df_all["category"].mode().iloc[0]
        hist = df_all[df_all["category"].astype(str) == str(cat_mode)]
        
    # Monthly seasonality mapping
    if "date" in df_all.columns:
        monthly_raw = df_all.groupby(df_all["date"].dt.month)["units_sold"].mean().fillna(0.0)
        overall_mean = monthly_raw.mean()
        if overall_mean > 0:
            monthly = monthly_raw / overall_mean
        else:
            monthly = pd.Series({m:1.0 for m in range(1,13)})
    else:
        monthly = pd.Series({m:1.0 for m in range(1,13)})
        
    if monthly.sum() == 0:
        monthly = pd.Series({m:1.0 for m in range(1,13)})
        
    # Initialize state variables for simulation using find_or_create_row for consistency
    initial_row = find_or_create_row(df_all, sku, channel, region)
    
    current_inventory = safe_float(initial_row.get("inventory_level"), 1000.0)
    current_comp_price = safe_float(initial_row.get("competitor_price"), 100.0)
    base_local_price = safe_float(initial_row.get("local_price"), 100.0)
    base_units = safe_float(initial_row.get("units_sold"), 100.0)
    
    # Simple holiday list (month, day) - Example Indian holidays
    holidays = [(1, 1), (1, 26), (8, 15), (10, 2), (12, 25), (11, 12)]
    
    results = []
    # PRE-CALCULATIONS FOR PERFORMANCE
    sku_str = str(sku)
    sku_data_filtered = df_all[df_all["sku_id"].astype(str) == sku_str]
    
    # 1. Pre-calculate elasticity once (constant for duration)
    elasticity = estimate_elasticity(df_all, sku=sku, category=initial_row.get("category"))
    
    # 2. Pre-calculate prev_median once (constant for duration)
    prev_median = None
    if not sku_data_filtered.empty:
        prev_median = float(sku_data_filtered["local_price"].median(skipna=True))
    
    if prev_median is None or math.isnan(prev_median):
        cat = str(initial_row.get("category", ""))
        ch = str(initial_row.get("channel", ""))
        rg = str(initial_row.get("region", ""))
        grp = df_all[
            (df_all["category"].astype(str) == cat) &
            (df_all["channel"].astype(str) == ch) &
            (df_all["region"].astype(str) == rg)
        ]
        if not grp.empty:
            prev_median = float(grp["local_price"].median(skipna=True))
        else:
            prev_median = float(df_all["local_price"].median(skipna=True))

    # Pre-instantiate agents and assembler
    promo_agent = PromoAgent(current_guardrails)
    comp_agent = CompetitorAgent()
    inv_agent = InventoryAgent()
    assembler = CandidateAssembler(current_guardrails)

    for i in range(days):
        d = start + pd.Timedelta(days=i)
        m = int(d.month)
        
        # Seasonality
        idx = monthly.get(m, monthly.mean())
        
        # Dynamic factors
        is_weekend = d.weekday() >= 5
        is_holiday = (d.month, d.day) in holidays
        
        # Festival lift
        daily_lift = 1.0
        if is_weekend: daily_lift *= 1.1
        if is_holiday: daily_lift *= 1.25
        
        # Promo logic
        is_promo = False
        if is_holiday or (is_weekend and random.random() < 0.3):
            is_promo = True
        elif random.random() < 0.05: 
            is_promo = True
            
        promo_depth = 0.0
        if is_promo:
            promo_depth = random.uniform(10.0, 25.0)
            
        # Competitor Price (Random Walk)
        drift = (safe_float(initial_row.get("competitor_price"), 100.0) - current_comp_price) * 0.1
        noise = current_comp_price * random.uniform(-0.02, 0.02)
        current_comp_price += drift + noise
        
        # Create synthetic row
        synthetic = initial_row.copy()
        synthetic["date"] = d
        synthetic["competitor_price"] = current_comp_price
        synthetic["promo_depth_pct"] = promo_depth
        synthetic["inventory_level"] = current_inventory
        synthetic["festival_lift"] = daily_lift
        synthetic["seasonality_index"] = float(idx)
        synthetic["local_price"] = base_local_price

        # Fast BasePrice logic
        b_neutral = prev_median * float(idx) * daily_lift
        base_candidates = {
            "conservative": round(b_neutral * 0.98, 2),
            "neutral": round(b_neutral, 2),
            "aggressive": round(b_neutral * 1.05, 2)
        }
        base_tr = AgentTrace("base", base_candidates, {"prev_median": prev_median, "seasonality": float(idx), "festival": daily_lift})
        
        # Run other agents (usually fast as they don't do lookups)
        promo_tr = promo_agent.run(synthetic, base_tr.candidates)
        comp_tr = comp_agent.run(synthetic, promo_tr.candidates)
        inv_tr = inv_agent.run(synthetic, comp_tr.candidates)
        
        traces = {"base": base_tr, "promo": promo_tr, "competitor": comp_tr, "inventory": inv_tr}
        mapped = assembler.assemble(traces)
        
        candidates_detail = {}
        
        # Helper to predict units
        def predict_units(p):
            hist_units = safe_float(synthetic.get("units_sold"), 0.0)
            hist_price = safe_float(synthetic.get("local_price"), p)
            if hist_units <= 0: hist_units = 1.0
            
            # Apply seasonality and lift
            base_demand = hist_units * idx * daily_lift
            
            # Apply elasticity
            units_new = base_demand * ((hist_price + 1e-9)/(p + 1e-9)) ** max(abs(elasticity), 0.1)
            return max(units_new, 0.0)

        for k, v in mapped.items():
            pred_u = predict_units(v)
            candidates_detail[k] = {"price": round(v,2), "pred_units": round(pred_u, 2), "elasticity": elasticity}
            
        sel = selector_and_sop(synthetic, mapped, current_guardrails)
        
        # Update inventory for next day
        units_sold_today = predict_units(sel["price_recommended"])
        current_inventory = max(0, current_inventory - units_sold_today)
        
        # Restock logic
        if current_inventory < 10:
             current_inventory += 500 
        
        results.append({
            "date": str(d.date()), 
            "seasonality_index": float(idx), 
            "traces": {k:v.candidates for k,v in traces.items()}, 
            "candidates": candidates_detail, 
            "selection": sel
        })
        
    return results

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
