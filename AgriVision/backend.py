# backend.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta
import json

app = FastAPI(title="AgriVision API", description="Crop Price Prediction and Farmer Recommendations")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_crop_data(crop_name):
    """Load data for a specific crop"""
    try:
        df = pd.read_csv(f"{crop_name}_prices.csv", parse_dates=["month"])
        df.set_index("month", inplace=True)
        return df
    except FileNotFoundError:
        return None

# Available crops
CROPS = {
    "tomato": {
        "name": "Tomato",
        "data": load_crop_data("tomato"),
        "unit": "₹/kg"
    },
    "rice": {
        "name": "Rice",
        "data": load_crop_data("rice"),
        "unit": "₹/kg"
    },
    "wheat": {
        "name": "Wheat",
        "data": load_crop_data("wheat"),
        "unit": "₹/kg"
    },
    "potato": {
        "name": "Potato",
        "data": load_crop_data("potato"),
        "unit": "₹/kg"
    },
    "onion": {
        "name": "Onion",
        "data": load_crop_data("onion"),
        "unit": "₹/kg"
    }
}

@app.get("/crops")
def get_crops():
    """Get list of available crops"""
    return {"crops": list(CROPS.keys())}

@app.get("/crop/{crop_name}")
def get_crop_data(crop_name: str):
    """Get historical data and analysis for a specific crop"""
    if crop_name not in CROPS:
        return {"error": "Crop not found"}
    
    crop_data = CROPS[crop_name]["data"]
    if crop_data is None:
        return {"error": f"Data not available for {crop_name}"}
    
    # Calculate statistics
    current_price = float(crop_data["price"].iloc[-1])
    highest_price = float(crop_data["price"].max())
    lowest_price = float(crop_data["price"].min())
    avg_price = float(crop_data["price"].mean())
    
    highest_date = crop_data["price"].idxmax().strftime("%Y-%m")
    lowest_date = crop_data["price"].idxmin().strftime("%Y-%m")
    
    recent_6m = crop_data["price"].tail(6).mean()
    previous_6m = crop_data["price"].tail(12).head(6).mean()
    trend_percentage = ((recent_6m - previous_6m) / previous_6m) * 100
    
    historical_data = []
    for date, price in crop_data["price"].items():
        historical_data.append({
            "month": date.strftime("%Y-%m"),
            "price": float(price)
        })
    
    return {
        "crop_name": CROPS[crop_name]["name"],
        "unit": CROPS[crop_name]["unit"],
        "current_price": current_price,
        "highest_price": highest_price,
        "lowest_price": lowest_price,
        "avg_price": avg_price,
        "highest_date": highest_date,
        "lowest_date": lowest_date,
        "trend_percentage": round(trend_percentage, 2),
        "historical_data": historical_data
    }

@app.get("/forecast/{crop_name}")
def forecast(crop_name: str, months: int = 6):
    """Generate price forecast and recommendations for a crop"""
    if crop_name not in CROPS:
        return {"error": "Crop not found"}
    
    crop_data = CROPS[crop_name]["data"]
    if crop_data is None:
        return {"error": f"Data not available for {crop_name}"}
    
    model = ARIMA(crop_data["price"], order=(1,1,1))
    model_fit = model.fit()
    forecast = model_fit.get_forecast(steps=months)
    predicted_mean = forecast.predicted_mean
    conf_int = forecast.conf_int()

    last_year_avg = crop_data["price"].tail(12).mean()
    current_price = float(crop_data["price"].iloc[-1])
    
    forecast_data = []
    for date, price in predicted_mean.items():
        lower = conf_int.loc[date].iloc[0]
        upper = conf_int.loc[date].iloc[1]
        
        price_ratio = price / last_year_avg
        current_trend = (crop_data["price"].tail(6).mean() - crop_data["price"].tail(12).head(6).mean()) / crop_data["price"].tail(12).head(6).mean()
        
        if current_trend < -0.05:  # Declining trend
            advice = "HOLD"
            reasoning = "Prices are declining. Hold your produce until market conditions improve."
            confidence = "High"
        elif price > 1.15 * last_year_avg:
            advice = "SELL NOW"
            reasoning = "Prices are above average. Good opportunity to sell for profit."
            confidence = "High"
        elif price > 1.05 * last_year_avg:
            advice = "SELL"
            reasoning = "Prices are slightly above average. Consider selling for moderate profit."
            confidence = "Medium"
        else:
            advice = "HOLD"
            reasoning = "Prices are near average. Hold until better market conditions."
            confidence = "Medium"
        
        forecast_data.append({
            "month": date.strftime("%Y-%m"),
            "price": float(price),
            "lower": float(lower),
            "upper": float(upper),
            "advice": advice,
            "reasoning": reasoning,
            "confidence": confidence
        })
    
    avg_forecast_price = predicted_mean.mean()
    current_trend = (crop_data["price"].tail(6).mean() - crop_data["price"].tail(12).head(6).mean()) / crop_data["price"].tail(12).head(6).mean()
    
    if current_trend < -0.05:  # Declining trend
        overall_advice = "HOLD - Prices are declining, wait for better market conditions"
    elif avg_forecast_price > current_price * 1.1:
        overall_advice = "SELL - Expected price increase, good time to sell"
    elif avg_forecast_price < current_price * 0.9:
        overall_advice = "HOLD - Expected price decrease, wait for recovery"
    else:
        overall_advice = "MONITOR - Prices expected to remain stable"
    
    return {
        "crop_name": CROPS[crop_name]["name"],
        "unit": CROPS[crop_name]["unit"],
        "current_price": current_price,
        "forecast_data": forecast_data,
        "overall_advice": overall_advice,
        "avg_forecast_price": float(avg_forecast_price)
    }

@app.get("/")
def root():
    return {"message": "AgriVision API - Crop Price Prediction and Farmer Recommendations"}
