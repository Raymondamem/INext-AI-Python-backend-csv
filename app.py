import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
from datetime import datetime
import json
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from openai import OpenAI
import logging
import asyncio
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import os
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Rate limiting configuration
limiter = Limiter(key_func=get_remote_address, default_limits=["1000/minute"])

# Lifecycle event for model loading
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load pre-trained model and encoder on startup
    global model, le
    model = joblib.load('emotion_predictor.pkl')
    le = joblib.load('label_encoder.pkl')
    yield
    # Cleanup on shutdown (optional)
    logger.info("Shutting down backend")

# Initialize FastAPI app with lifespan and limiter
app = FastAPI(lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "your_openai_api_key_here"))

# Pydantic models
class Trade(BaseModel):
    user_id: int
    symbol: str
    type: str
    side: str
    amount: float

class Mood(BaseModel):
    user_id: int
    mood: str
    timestamp: str

class JournalEntry(BaseModel):
    user_id: int
    entry: str
    timestamp: str

# In-memory storage (replace with database like MongoDB or ICP canister in production)
users = {}
trades = {}
moods = {}
journal = {}

# Mock market data (replace with real API in production)
market_data = {
    "BTCUSDT": [{"timestamp": 1721753783000, "open": 60000, "high": 60500, "low": 59500, "close": 60200, "volume": 1000}],
    "ETHUSDT": [{"timestamp": 1721753783000, "open": 3000, "high": 3050, "low": 2950, "close": 3020, "volume": 500}],
}

# Feature engineering function
def engineer_features(df):
    df['time_diff'] = df['timestamp'].diff().dt.total_seconds() / 60
    df['slippage_factor'] = 0.99
    df['price_in'] = (df['volume_usd'] / df['amount_in']) * df['slippage_factor']
    df['price_out'] = (df['volume_usd'] / df['amount_out'].replace('INVALID', 42490798.02)) * df['slippage_factor']
    df['market_price'] = df['price_out'].rolling(window=60).mean()
    df['price_change_pct'] = ((df['price_out'] - df['market_price']) / df['market_price']) * 100
    df['account_equity'] = 1000
    df['leverage'] = (df['amount_in'] * df['price_in']) / df['account_equity']
    df['position_change'] = df['leverage'].pct_change()
    df['trade_pair'] = df['in_token'] + '_' + df['out_token']
    df['entry_price'] = df.groupby('trade_pair')['price_in'].shift(1)
    df['exit_price'] = df['price_out']
    df['pnl'] = (df['exit_price'] - df['entry_price']) * df['amount_out'] * 0.997
    df['is_win'] = df['pnl'] > 0
    df['is_loss'] = df['pnl'] < 0
    df['win_streak'] = (df['pnl'] > 0).groupby((df['pnl'] <= 0).cumsum()).cumcount() + 1
    df['loss_streak'] = (df['pnl'] < 0).groupby((df['pnl'] >= 0).cumsum()).cumcount() + 1
    df['consecutive_losses'] = (df['pnl'] < 0).rolling(window=3).sum()
    df['consecutive_wins'] = (df['pnl'] > 0).rolling(window=3).sum()
    return df

# Emotion detection
def detect_emotion(df):
    df['emotion'] = 'neutral'
    df['trigger_details'] = None
    for i in range(1, len(df)):
        prev_row, curr_row = df.iloc[i-1], df.iloc[i]
        if (curr_row['price_change_pct'] > 20 and curr_row['time_diff'] < 30 and curr_row['win_streak'] > 1):
            df.loc[i, 'emotion'] = 'fomo'
            df.loc[i, 'trigger_details'] = f"Price spike: {curr_row['price_change_pct']:.2f}%, win streak: {curr_row['win_streak']}"
        elif (curr_row['time_diff'] < 2 and prev_row['consecutive_losses'] >= 2):
            df.loc[i, 'emotion'] = 'revenge'
            df.loc[i, 'trigger_details'] = f"After {prev_row['consecutive_losses']} losses, latency: {curr_row['time_diff']}"
        elif (curr_row['position_change'] > 0.5 and prev_row['consecutive_wins'] >= 2):
            df.loc[i, 'emotion'] = 'greed'
            df.loc[i, 'trigger_details'] = f"After {prev_row['consecutive_wins']} wins, pos change: {curr_row['position_change']:.2f}"
        elif (curr_row['time_diff'] < 1 and curr_row['pnl'] > 0 and abs(curr_row['pnl']) < 10):
            df.loc[i, 'emotion'] = 'fear'
            df.loc[i, 'trigger_details'] = f"Early close: {curr_row['pnl']:.2f}, latency: {curr_row['time_diff']}"
    return df

# Prepare ML data from last 3 trades
def prepare_ml_data(df):
    features = ['time_diff', 'price_change_pct', 'position_change', 'consecutive_wins', 'consecutive_losses', 'win_streak', 'loss_streak']
    ml_data = []
    for i in range(2, len(df) - 1):
        prev3 = df.iloc[i-2:i].copy()
        curr = df.iloc[i]
        if len(prev3) == 3:
            ml_data.append({
                'time_diff': prev3['time_diff'].mean(),
                'price_change_pct': prev3['price_change_pct'].mean(),
                'position_change': prev3['position_change'].mean(),
                'consecutive_wins': prev3['consecutive_wins'].iloc[-1],
                'consecutive_losses': prev3['consecutive_losses'].iloc[-1],
                'win_streak': prev3['win_streak'].iloc[-1],
                'loss_streak': prev3['loss_streak'].iloc[-1],
                'emotion': curr['emotion']
            })
    return pd.DataFrame(ml_data) if ml_data else pd.DataFrame()

# Warning and recommendation generation
async def get_emotion_warning(wallet_df, predicted_emotion):
    recent_trades = wallet_df.tail(3).to_dict(orient='records')
    prompt = f"""
    User with wallet {wallet_df['wallet'].iloc[0]} has shown a predicted emotion of {predicted_emotion} based on their last 3 trades. 
    Trade data: {json.dumps(recent_trades, default=str)}
    Provide insight, warning, recommendation, and advice in JSON format.
    """
    try:
        response = await asyncio.to_thread(
            lambda: client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                timeout=30
            )
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"API error: {e}")
        return {"insight": "Error fetching insight", "warning": "Unable to warn", "recommendation": "none", "advice": "Contact support"}

# Endpoints

@app.post('/api/place_order')
# @limiter.limit("50/minute")
async def place_order(trade: Trade):
    user_id = trade.user_id
    trade_data = trade.dict() | {"timestamp": datetime.utcnow().isoformat(), "volume_usd": trade.amount, "wallet": users.get(user_id, {}).get('wallet', f"0x{user_id:x}...")}
    if user_id not in trades:
        trades[user_id] = []
    trades[user_id].append(trade_data)

    # Engineer features and detect emotion
    df = pd.DataFrame(trades[user_id])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = engineer_features(df)
    df = detect_emotion(df)

    # Save vector
    features = df[['time_diff', 'price_change_pct', 'position_change', 'consecutive_wins', 'consecutive_losses', 'win_streak', 'loss_streak']].to_numpy()
    joblib.dump(features, f'user_{user_id}_features.pkl')

    # Predict next emotion
    ml_df = prepare_ml_data(df)
    if not ml_df.empty:
        X = ml_df.drop(columns=['emotion'])
        y = le.transform(ml_df['emotion'])
        model.fit(X, y)  # Retrain with new data
        joblib.dump(model, 'emotion_predictor.pkl')
        joblib.dump(le, 'label_encoder.pkl')

        if len(df) >= 4:
            predicted_emotion, _ = predict_next_emotion(df, model, le)
            warning = await get_emotion_warning(df, predicted_emotion)
            return JSONResponse({"status": "success", "trade_id": len(trades[user_id]), "predicted_emotion": predicted_emotion, "warning": warning})

    return JSONResponse({"status": "success", "trade_id": len(trades[user_id])})

@app.get('/api/market_data/{symbol}')
# @limiter.limit("100/minute")
async def get_market_data(symbol: str, timeframe: str = "5m"):
    if symbol not in market_data:
        raise HTTPException(status_code=404, detail="Symbol not found")
    return JSONResponse(market_data[symbol])

@app.post('/mood/')
# @limiter.limit("20/minute")
async def submit_mood(mood: Mood):
    user_id = mood.user_id
    if user_id not in moods:
        moods[user_id] = []
    moods[user_id].append(mood.dict())

    # Update emotional trends
    df = pd.DataFrame(moods[user_id])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    trends = df.groupby('mood').size().to_dict()
    return JSONResponse({"status": "success", "trends": trends})

@app.get('/dashboard/{user_id}/emotional_trends')
# @limiter.limit("50/minute")
async def get_emotional_trends(user_id: int):
    if user_id not in moods:
        return JSONResponse({"moods": [], "indices": {}})
    df = pd.DataFrame(moods[user_id])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    recent_moods = df.tail(5).to_dict(orient='records')
    indices = {"fomo_index": 0, "greed_index": 0, "fear_index": 0, "revenge_index": 0, "confidence_index": 0}  # Mock; compute from trades
    return JSONResponse({"moods": recent_moods, "indices": indices})

@app.get('/users/{user_id}')
# @limiter.limit("50/minute")
async def get_user(user_id: int):
    if user_id not in users:
        users[user_id] = {"username": f"User{user_id}", "xp": 0, "level": 1, "wallet": f"0x{user_id:x}...", "wishlist": "None"}
    return JSONResponse(users[user_id])

@app.post('/journal/')
# @limiter.limit("20/minute")
async def submit_journal(entry: JournalEntry):
    user_id = entry.user_id
    if user_id not in journal:
        journal[user_id] = []
    journal[user_id].append(entry.dict())
    return JSONResponse({"status": "success"})

@app.get('/recommendations/{user_id}')
# @limiter.limit("50/minute")
async def get_recommendations(user_id: int):
    if user_id not in trades or len(trades[user_id]) < 4:
        return JSONResponse([])
    df = pd.DataFrame(trades[user_id])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = engineer_features(df)
    df = detect_emotion(df)
    predicted_emotion, _ = predict_next_emotion(df, model, le)
    warning = await get_emotion_warning(df, predicted_emotion)
    recommendations = [
        {"timestamp": datetime.utcnow().isoformat(), "message": f"Warning: {warning['warning']}", "severity": "warning"},
        {"timestamp": datetime.utcnow().isoformat(), "message": f"Recommendation: {warning['recommendation']}", "severity": "info"}
    ]
    return JSONResponse(recommendations)

# WebSocket for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(json.dumps(market_data["BTCUSDT"][-1]))  # Mock real-time data
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")

# Helper function for emotion prediction
def predict_next_emotion(df, model, le):
    if len(df) < 4:
        return "neutral", "Not enough trades"
    last3 = df.iloc[-3:].copy()
    features = {
        'time_diff': last3['time_diff'].mean(),
        'price_change_pct': last3['price_change_pct'].mean(),
        'position_change': last3['position_change'].mean(),
        'consecutive_wins': last3['consecutive_wins'].iloc[-1],
        'consecutive_losses': last3['consecutive_losses'].iloc[-1],
        'win_streak': last3['win_streak'].iloc[-1],
        'loss_streak': last3['loss_streak'].iloc[-1]
    }
    X_pred = pd.DataFrame([features])
    pred_encoded = model.predict(X_pred)[0]
    emotion = le.inverse_transform([pred_encoded])[0]
    return emotion, "Prediction based on last 3 trades"

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)