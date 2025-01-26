# 必要なライブラリをインストール
from flask import Flask, request, render_template
import yfinance as yf
from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import os
import time

# Flaskアプリケーションの作成
app = Flask(__name__, template_folder="templates")

# 株価データを取得する関数
def get_stock_data(ticker, start_date, end_date):
    for i in range(3):  # 最大3回試行
        try:
            data = yf.download(ticker, start=start_date, end=end_date)
            if not data.empty:
                break
        except Exception as e:
            print(f"Retry {i+1}: Failed to fetch data for {ticker}. Error: {e}")
            time.sleep(2)  # 2秒待機
    else:
        raise ValueError(f"Failed to download data for {ticker} after multiple attempts.")
    
    data = data[['Close']].reset_index()
    data.columns = ['ds', 'y']
    data['ds'] = data['ds'].dt.tz_localize(None)  # タイムゾーンを削除
    return data


# モデル構築と予測を行う関数
def forecast_stock(data, periods):
    model = Prophet()
    model.fit(data)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return model, forecast

# グラフを画像として返す関数
def plot_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    return img_base64

# ルートページ
@app.route("/", methods=["GET", "POST"])
def index():
    print("Current working directory:", os.getcwd())  # デバッグ用
    if not os.path.exists("templates"):
        print("Templates folder not found!")  # デバッグ用
        return "Templates folder not found!", 500

    if "index.html" not in os.listdir("templates"):
        print("index.html not found in templates folder!")  # デバッグ用
        return "index.html not found in templates folder!", 500

    forecast_plot = None
    if request.method == "POST":
        ticker = request.form.get("ticker")
        start_date = request.form.get("start_date")
        end_date = request.form.get("end_date")
        forecast_days = int(request.form.get("forecast_days"))

        stock_data = get_stock_data(ticker, start_date, end_date)
        model, forecast = forecast_stock(stock_data, forecast_days)
        fig = model.plot(forecast)
        forecast_plot = plot_to_base64(fig)

    return render_template("index.html", forecast_plot=forecast_plot)

if __name__ == "__main__":
    app.run(debug=True)
