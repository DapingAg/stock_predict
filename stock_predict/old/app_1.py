from flask import Flask, request, render_template
import yfinance as yf
from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__, template_folder="templates")

# 株価データを取得する関数
def get_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        raise ValueError(f"No data found for ticker {ticker} in the given date range.")
    data = data[['Close']].reset_index()
    data.columns = ['ds', 'y']
    data['ds'] = data['ds'].dt.tz_localize(None)
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
    # 軸タイトルを設定
    ax = fig.gca()
    ax.set_xlabel("Day")
    ax.set_ylabel("Price")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    return img_base64

@app.route("/", methods=["GET", "POST"])
def index():
    plots = []  # 各銘柄の予測グラフ
    stats = []  # 各銘柄の統計情報
    error_message = None

    if request.method == "POST":
        tickers = request.form.get("tickers").split(",")  # カンマ区切りで複数銘柄を取得
        start_date = request.form.get("start_date")
        end_date = request.form.get("end_date")
        forecast_days = int(request.form.get("forecast_days"))

        for ticker in tickers[:4]:  # 最大4つの銘柄を処理
            ticker = ticker.strip()
            try:
                # データ取得と予測
                stock_data = get_stock_data(ticker, start_date, end_date)
                model, forecast = forecast_stock(stock_data, forecast_days)

                # グラフ作成
                fig = model.plot(forecast)
                plots.append((ticker, plot_to_base64(fig)))

                # 統計データ作成
                forecast_stats = {
                    "ticker": ticker,
                    "mean": forecast['yhat'].mean(),
                    "median": forecast['yhat'].median(),
                    "max": forecast['yhat'].max(),
                    "min": forecast['yhat'].min(),
                    "std_dev": forecast['yhat'].std()
                }
                stats.append(forecast_stats)

            except Exception as e:
                error_message = str(e)

    return render_template("index.html", plots=plots, stats=stats, error_message=error_message)

if __name__ == "__main__":
    app.run(debug=True)
