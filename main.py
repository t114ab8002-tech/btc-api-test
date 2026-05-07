import yfinance as yf
import pandas as pd
import numpy as np
import ta
import pickle
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# 全域變數，一開始先空著省記憶體
MODEL = None
SCALER = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def lazy_load():
    """只有在需要時才載入模型，節省啟動時的記憶體壓力"""
    global MODEL, SCALER
    
    # 1. 處理模型載入
    if MODEL is None:
        try:
            # 在函式內部才匯入 tensorflow，避免一開機就爆記憶體
            from tensorflow.keras.models import load_model
            # .h5 格式建議只使用 compile=False 即可
            MODEL = load_model("lstm_best.h5", compile=False)
            print("模型載入成功！")
        except Exception as e:
            print(f"模型載入失敗: {e}")
            raise e # 拋出錯誤讓 API 能夠回傳錯誤訊息
            
    # 2. 處理 Scaler 載入 (這部分要獨立出來，不能縮排在 except 裡面)
    if SCALER is None:
        try:
            with open("lstm_scaler.pkl", "rb") as f:
                SCALER = pickle.load(f)
            print("Scaler 載入成功！")
        except Exception as e:
            print(f"Scaler 載入失敗: {e}")
            raise e

FEATURES = [
    "Close", "High", "Low", "Open", "Volume",
    "trend_macd", "trend_macd_signal", "trend_macd_diff",
    "momentum_rsi", "volatility_bbm", "volatility_bbh", 
    "volatility_bbl", "volatility_atr", "trend_adx", 
    "trend_ichimoku_a", "trend_ichimoku_b",
    "momentum_stoch_rsi", "momentum_stoch_rsi_k", 
    "momentum_stoch_rsi_d", "volume_obv"
]

@app.get("/api/get_latest_prediction")
def predict():
    try:
        lazy_load() # 呼叫時才載入
        
        # 抓取資料
        df = yf.download("BTC-USD", period="60d", interval="1d")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # 計算指標
        df = ta.add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)
        target_data = df[FEATURES].tail(20)
        
        # 標準化與預測
        scaled_data = SCALER.transform(target_data.values)
        input_tensor = np.array([scaled_data])
        
        prediction = MODEL.predict(input_tensor)
        prob = float(prediction.flatten()[0])
        
        return {"status": "success", "probability": round(prob * 100, 2)}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/")
def home():
    return {"message": "AI Brain is Online"}
