from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import Cookie, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.sessions import SessionMiddleware

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "sell_smart.db"
BUNDLE_PATH = Path(os.getenv("SELL_SMART_BUNDLE", DATA_DIR / "Sell_Smart_AI_Decision_System_FINAL.joblib"))
SESSION_SECRET = os.getenv("SESSION_SECRET", "change-this-for-production-" + secrets.token_hex(32))

if not BUNDLE_PATH.exists():
    raise RuntimeError(f"Missing authoritative decision artifact: {BUNDLE_PATH}")

# The artifact contains notebook __main__ references. The placeholders allow joblib to
# resolve those names; we do not invoke the serialized placeholder functions.
import __main__ as _main
for _name in [
    "storage_loss_prediction", "estimator_feature_names", "predict_price",
    "predict_transport", "sell_smart_decision", "price_prediction_function",
    "transport_prediction_function", "storage_loss_function", "decision_function",
]:
    if not hasattr(_main, _name):
        setattr(_main, _name, lambda *args, **kwargs: None)

try:
    BUNDLE = joblib.load(BUNDLE_PATH)
except Exception as exc:
    raise RuntimeError(
        "Sell Smart artifact could not be loaded. Deploy with scikit-learn==1.6.1 "
        "and the pinned requirements."
    ) from exc

if not isinstance(BUNDLE, dict):
    raise RuntimeError("Authoritative artifact must load as a dictionary")

price_artifact = BUNDLE["price_model"]
storage_artifact = BUNDLE["storage_model"]
transport_artifact = BUNDLE["transport_model"]


def unwrap_estimator(artifact: Any) -> Any:
    if hasattr(artifact, "predict"):
        return artifact
    if isinstance(artifact, dict):
        for key in ["model", "estimator", "pipeline", "regressor", "price_model"]:
            value = artifact.get(key)
            if hasattr(value, "predict"):
                return value
    return None


price_model = unwrap_estimator(price_artifact)
transport_model = unwrap_estimator(transport_artifact)


def estimator_feature_names(estimator: Any):
    names = getattr(estimator, "feature_names_in_", None)
    return list(names) if names is not None else None


def predict_price(price_features: dict[str, Any], current_price: float) -> float:
    if price_model is None:
        raise RuntimeError("No price estimator available")
    names = estimator_feature_names(price_model)
    if names is not None:
        missing = [f for f in names if f not in price_features]
        if missing:
            raise ValueError(f"Missing price-model features: {missing}")
        X = pd.DataFrame([{f: price_features[f] for f in names}])
    else:
        X = pd.DataFrame([price_features])

    # The authoritative artifact predicts target_log_return. Its metadata documents
    # a Naive + Extra Trees ensemble where the naive baseline is the current price
    # (zero log-return) and the Extra Trees forecast receives the stored weight.
    raw_log_return = float(np.asarray(price_model.predict(X)).reshape(-1)[0])
    weight = float(price_artifact.get("ensemble_extra_trees_weight", 0.19)) if isinstance(price_artifact, dict) else 0.19
    forecast_method = price_artifact.get("forecast_method") if isinstance(price_artifact, dict) else None
    target_name = price_artifact.get("target") if isinstance(price_artifact, dict) else None
    if target_name == "target_log_return" or forecast_method == "current_price_times_exp_predicted_log_return":
        effective_log_return = weight * raw_log_return
        return float(current_price * np.exp(effective_log_return))
    return raw_log_return


def predict_transport(transport_features: dict[str, Any]) -> float:
    if transport_model is None:
        raise RuntimeError("No transport estimator available")
    names = estimator_feature_names(transport_model)
    if names is not None:
        missing = [f for f in names if f not in transport_features]
        if missing:
            raise ValueError(f"Missing transport-model features: {missing}")
        X = pd.DataFrame([{f: transport_features[f] for f in names}])
    else:
        X = pd.DataFrame([transport_features])
    return float(np.asarray(transport_model.predict(X)).reshape(-1)[0])


def storage_loss_prediction(region: str):
    if not isinstance(storage_artifact, dict):
        raise TypeError("Expected storage artifact to be a dictionary")
    region_stats = storage_artifact.get("region_stats", {})
    global_mean = float(storage_artifact.get("global_mean", 0.0))
    alpha = float(storage_artifact.get("region_smoothing_alpha", 50.0))
    stats = region_stats.get(region) if isinstance(region_stats, dict) else None
    if stats is None and isinstance(region_stats, dict):
        stats = region_stats.get(str(region))
    if stats is None:
        return global_mean, "global_mean_fallback"
    if isinstance(stats, dict):
        region_mean = stats.get("mean", stats.get("loss_mean", stats.get("storage_loss_mean")))
        region_count = stats.get("count", stats.get("n", stats.get("region_count")))
    elif isinstance(stats, (list, tuple)) and len(stats) >= 2:
        region_mean, region_count = stats[0], stats[1]
    else:
        region_mean = region_count = None
    if region_mean is None or region_count is None:
        return global_mean, "global_mean_fallback"
    return (
        (float(region_mean) * float(region_count) + global_mean * alpha)
        / (float(region_count) + alpha),
        "region_smoothed",
    )


def sell_smart_decision(
    current_price: float,
    quantity: float,
    region: str,
    price_features: dict[str, Any],
    transport_features: dict[str, Any],
    transport_cost_now: float | None = None,
    transport_cost_later: float | None = None,
    storage_cost_per_unit: float = 0.0,
):
    current_price = float(current_price)
    quantity = float(quantity)
    predicted_price = predict_price(price_features, current_price)
    storage_loss_pct, storage_status = storage_loss_prediction(region)
    if transport_cost_now is None:
        transport_cost_now = predict_transport(transport_features)
    if transport_cost_later is None:
        transport_cost_later = float(transport_cost_now)
    loss_fraction = max(0.0, storage_loss_pct) / 100.0
    sale_quantity_later = quantity * (1.0 - loss_fraction)
    sell_now_value = quantity * current_price - float(transport_cost_now)
    store_then_sell_value = (
        sale_quantity_later * predicted_price
        - float(transport_cost_later)
        - quantity * float(storage_cost_per_unit)
    )
    advantage = store_then_sell_value - sell_now_value
    base_value = max(abs(sell_now_value), 1e-9)
    advantage_pct = 100.0 * advantage / base_value
    threshold = float(BUNDLE.get("decision_threshold_pct", 2.0))
    if advantage_pct >= threshold:
        decision = "STORE"
    elif advantage_pct <= -threshold:
        decision = "SELL_NOW"
    else:
        decision = "STORE_CAUTION"
    if decision == "STORE":
        recommendation = "Store the wheat if the real storage conditions match the estimate, then consider selling as the market approaches the predicted price."
    elif decision == "SELL_NOW":
        recommendation = "Selling now is financially preferable under the model assumptions; avoid unnecessary storage costs and losses."
    else:
        recommendation = "The difference is small. Compare actual local offers, storage conditions, and transport costs before deciding."
    return {
        "current_price": current_price,
        "predicted_price": predicted_price,
        "expected_storage_loss_pct": float(storage_loss_pct),
        "storage_status": storage_status,
        "transport_cost_now": float(transport_cost_now),
        "transport_cost_later": float(transport_cost_later),
        "quantity": quantity,
        "expected_quantity_after_storage": sale_quantity_later,
        "sell_now_value": sell_now_value,
        "store_then_sell_value": store_then_sell_value,
        "expected_financial_advantage": advantage,
        "expected_financial_advantage_pct": advantage_pct,
        "decision": decision,
        "recommendation": recommendation,
        "reasons": [
            f"Predicted future price: {predicted_price:,.2f} ETB per unit.",
            f"Expected storage loss: {storage_loss_pct:.2f}%.",
            f"Selling now value after current transport: {sell_now_value:,.2f} ETB.",
            f"Store-then-sell estimated value: {store_then_sell_value:,.2f} ETB.",
            f"Estimated advantage of storing: {advantage:,.2f} ETB ({advantage_pct:.2f}%).",
        ],
    }


def db():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with db() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            phone TEXT,
            region TEXT,
            zone TEXT,
            woreda TEXT,
            created_at TEXT NOT NULL
        )
        """)
        conn.commit()


def hash_password(password: str, salt_hex: str | None = None):
    salt = bytes.fromhex(salt_hex) if salt_hex else secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 240_000)
    return digest.hex(), salt.hex()


def verify_password(password: str, stored: str, salt: str):
    candidate, _ = hash_password(password, salt)
    return hmac.compare_digest(candidate, stored)


def current_user(request: Request):
    uid = request.session.get("user_id")
    if not uid:
        return None
    with db() as conn:
        row = conn.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
    if not row:
        return None
    return {
        "id": row["id"],
        "full_name": row["full_name"],
        "email": row["email"],
        "phone": row["phone"],
        "region": row["region"],
        "zone": row["zone"],
        "woreda": row["woreda"],
        "created_at": row["created_at"],
    }


REGIONS = ["Addis Ababa", "Amhara", "Dire Dawa", "Oromia", "SNNPR", "Somali", "Tigray"]
MARKETS = [
    "Abi Adi","Addis Ababa","Adigrat","Adwa","Ajeber","Alaba","Alamata","Aleta Wondo","Amaro","Ambo","Arba Minch","Aroresa","Assela","Baher Dar","Beddenno","Bedessa","Bure","Dalocha","Debark","Debre Birhan","Debre Markos","Deder","Delo","Derashe","Dessie","Dila","Diredawa","Ebinat","Enseno","Fedis","Gode","Gonder","Gordamole","Hawassa","Hawzien","Hossana","Jijiga","Jimma","Kersa","Korem"
]
TRANSPORT_MODES = {1: "On foot / pack animal", 2: "Cart", 3: "Vehicle"}

def to_model_region(region: str | None):
    if not region: return "Oromia"
    direct = next((r for r in REGIONS if r.lower() == region.lower()), None)
    if direct: return direct
    if region in {"Sidama", "Central Ethiopia", "SNNP"}: return "SNNPR"
    if region == "Afar": return "Amhara"
    return "Oromia"


def price_features(current, lag1, lag2, lag3, month, market, region):
    rolling3 = (current + lag1 + lag2) / 3
    rolling6 = (current + lag1 + lag2 + lag3) / 4
    pct = lambda a,b: ((a-b)/b*100) if b else 0
    angle = 2*np.pi*month/12
    return {
        "wheat_price_etb_per_tonne": current,
        "lag_1_price": lag1,
        "lag_2_price": lag2,
        "lag_3_price": lag3,
        "rolling_3mo_price": rolling3,
        "rolling_6mo_price": rolling6,
        "current_vs_lag1_pct": pct(current, lag1),
        "current_vs_roll3_pct": pct(current, rolling3),
        "current_vs_roll6_pct": pct(current, rolling6),
        "lag1_vs_lag2_pct": pct(lag1, lag2),
        "month_sin": np.sin(angle),
        "month_cos": np.cos(angle),
        "market": market,
        "region": region,
    }


class PredictRequest(BaseModel):
    current_price: float = Field(gt=0)
    quantity: float = Field(gt=0)
    market: str
    region: str
    lag1: float = Field(gt=0)
    lag2: float = Field(gt=0)
    lag3: float = Field(gt=0)
    harvest_kg: float = Field(gt=0)
    transport_mode: int = Field(ge=1, le=3)
    storage_cost_per_unit: float = Field(ge=0)


app = FastAPI(title="Sell Smart Presentation App", version="FINAL_V1")
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET, session_cookie="sell_smart_session", https_only=os.getenv("COOKIE_SECURE", "0") == "1", same_site="lax", max_age=60*60*24*7)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
init_db()


@app.get("/", response_class=HTMLResponse)
def index():
    return (BASE_DIR / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/api/me")
def me(request: Request):
    user = current_user(request)
    return {"authenticated": bool(user), "user": user}


@app.post("/api/signup")
def signup(request: Request, full_name: str = Form(...), email: str = Form(...), password: str = Form(...), phone: str = Form(""), region: str = Form(""), zone: str = Form(""), woreda: str = Form("")):
    if len(password) < 8: raise HTTPException(400, "Password must be at least 8 characters")
    ph, salt = hash_password(password)
    try:
        with db() as conn:
            cur = conn.execute("INSERT INTO users(full_name,email,password_hash,salt,phone,region,zone,woreda,created_at) VALUES(?,?,?,?,?,?,?,?,?)", (full_name.strip(), email.strip().lower(), ph, salt, phone.strip(), region.strip(), zone.strip(), woreda.strip(), datetime.now(timezone.utc).isoformat()))
            uid = cur.lastrowid
            conn.commit()
    except sqlite3.IntegrityError:
        raise HTTPException(409, "An account with that email already exists")
    request.session["user_id"] = uid
    return {"ok": True}


@app.post("/api/login")
def login(request: Request, email: str = Form(...), password: str = Form(...)):
    with db() as conn:
        row = conn.execute("SELECT * FROM users WHERE email=?", (email.strip().lower(),)).fetchone()
    if not row or not verify_password(password, row["password_hash"], row["salt"]):
        raise HTTPException(401, "Invalid email or password")
    request.session["user_id"] = row["id"]
    return {"ok": True}


@app.post("/api/logout")
def logout(request: Request):
    request.session.clear()
    return {"ok": True}


@app.put("/api/profile")
def update_profile(request: Request, full_name: str = Form(...), phone: str = Form(""), region: str = Form(""), zone: str = Form(""), woreda: str = Form("")):
    user = current_user(request)
    if not user: raise HTTPException(401, "Login required")
    with db() as conn:
        conn.execute("UPDATE users SET full_name=?, phone=?, region=?, zone=?, woreda=? WHERE id=?", (full_name.strip(), phone.strip(), region.strip(), zone.strip(), woreda.strip(), user["id"]))
        conn.commit()
    return {"ok": True}


@app.post("/api/predict")
def api_predict(request: Request, payload: PredictRequest):
    user = current_user(request)
    if not user: raise HTTPException(401, "Login required")
    model_region = to_model_region(payload.region)
    month = datetime.now().month
    pf = price_features(payload.current_price, payload.lag1, payload.lag2, payload.lag3, month, payload.market, model_region)
    sold_kg = payload.quantity * 1000
    tf = {
        "region_code": REGIONS.index(model_region) + 1 if model_region in REGIONS else 4,
        "zone_code": 1,
        "woreda_code": 1,
        "rural": 1,
        "harvest_kg": payload.harvest_kg,
        "sold_kg": sold_kg,
        "transport_mode_code": payload.transport_mode,
        "sale_month": month,
        "sale_year": datetime.now().year,
        "transactions": 1,
    }
    try:
        return sell_smart_decision(payload.current_price, payload.quantity, model_region, pf, tf, storage_cost_per_unit=payload.storage_cost_per_unit)
    except Exception as exc:
        raise HTTPException(500, f"Prediction failed: {exc}") from exc


@app.get("/api/meta")
def meta():
    return {"regions": REGIONS, "markets": MARKETS, "transport_modes": [{"code": k, "label": v} for k,v in TRANSPORT_MODES.items()], "artifact_version": BUNDLE.get("artifact_version"), "threshold": float(BUNDLE.get("decision_threshold_pct",2.0)), "sklearn": "1.6.1"}


@app.get("/api/health")
def health():
    return {"status":"ok", "artifact_version": BUNDLE.get("artifact_version"), "decision_threshold_pct": float(BUNDLE.get("decision_threshold_pct",2.0)), "sklearn_required":"1.6.1"}
