import yfinance as yf
import pandas as pd
import numpy as np
import ta
import pickle
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model

app = FastAPI()

# 1. 允許跨域連線（讓你的手機 APP 可以順利抓到資料）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. 載入大腦 (模型) 與 翻譯機 (Scaler)
# 請確保這兩個檔案已經上傳到 GitHub 同目錄下
MODEL = load_model("lstm_best.keras")
with open("lstm_scaler.pkl", "rb") as f:
    SCALER = pickle.load(f)

# 3. 定義特徵清單（必須與你訓練時用的順序、欄位完全一致）
FEATURES = [
    "Close", "High", "Low", "Open", "Volume",
    "trend_macd", "trend_macd_signal", "trend_macd_diff",
    "momentum_rsi", "volatility_bbm", "volatility_bbh", 
    "volatility_bbl", "volatility_atr", "trend_adx", 
    "trend_ichimoku_a", "trend_ichimoku_b",
    "momentum_stoch_rsi", "momentum_stoch_rsi_k", 
    "momentum_stoch_rsi_d", "volume_obv"
]

def get_realtime_data():
    """自動抓取最新市況並計算指標"""
    # 抓取最近 60 天資料以計算技術指標
    df = yf.download("BTC-USD", period="60d", interval="1d")
    
    # 處理 Yahoo Finance 的多重索引欄位
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # 使用 ta 套件計算所有技術指標
    df = ta.add_all_ta_features(
        df, open="Open", high="High", low="Low", 
        close="Close", volume="Volume", fillna=True
    )
    
    # 只拿最近 20 天 (對應你的 TIME_STEPS) 並過濾出模型需要的特徵
    target_data = df[FEATURES].tail(20)
    return target_data

@app.get("/api/get_latest_prediction")
def predict():
    try:
        # A. 抓取今日最新資料
        data_df = get_realtime_data()
        
        # B. 資料標準化 (使用你上傳的那個 scaler)
        scaled_data = SCALER.transform(data_df.values)
        
        # C. 轉成 LSTM 需要的 3D 形狀 (1, 20, 特徵數)
        input_tensor = np.array([scaled_data])
        
        # D. 讓模型進行推論
        prediction = MODEL.predict(input_tensor)
        prob = float(prediction[0][0])
        
        return {
            "status": "success",
            "probability": round(prob * 100, 2),
            "message": "預測已根據今日最新市況更新！"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/")
def home():
    return {"message": "比特幣 AI 雲端大腦運行中"}
