import yfinance as yf
import pandas as pd
import numpy as np
import ta
import pickle
import tensorflow as tf
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model

app = FastAPI()

# 1. 允許跨域連線
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. 載入大腦 (模型) 與 翻譯機 (Scaler)
# 使用 compile=False 和 safe_mode=False 強制跳過版本格式檢查
try:
    MODEL = load_model("lstm_best.keras", compile=False, safe_mode=False)
    print("模型載入成功！")
except Exception as e:
    print(f"模型載入失敗，嘗試另一種方式: {e}")
    # 備用方案：如果還是不行，這是在某些環境下的特殊載入法
    MODEL = tf.keras.models.load_model("lstm_best.keras", compile=False)

with open("lstm_scaler.pkl", "rb") as f:
    SCALER = pickle.load(f)

# 3. 定義特徵清單
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
    df = yf.download("BTC-USD", period="60d", interval="1d")
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df = ta.add_all_ta_features(
        df, open="Open", high="High", low="Low", 
        close="Close", volume="Volume", fillna=True
    )
    
    target_data = df[FEATURES].tail(20)
    return target_data

@app.get("/api/get_latest_prediction")
def predict():
    try:
        # A. 抓取今日最新資料
        data_df = get_realtime_data()
        
        # B. 資料標準化
        scaled_data = SCALER.transform(data_df.values)
        
        # C. 轉成 LSTM 需要的 3D 形狀
        input_tensor = np.array([scaled_data])
        
        # D. 讓模型進行推論 (使用 flatten 確保相容舊版 TF 的輸出格式)
        prediction = MODEL.predict(input_tensor)
        prob = float(prediction.flatten()[0])
        
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
