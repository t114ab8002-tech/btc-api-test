from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# 解決跨域問題 (CORS)，讓你的前端可以順利拿到資料
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "歡迎來到比特幣預測 API 伺服器！"}

@app.get("/api/get_latest_prediction")
def get_prediction():
    return {
        "status": "success",
        "probability": 88.5,
        "message": "這是來自 Render 雲端主機的即時資料！"
    }