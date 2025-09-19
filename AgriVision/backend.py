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

def load_crop_data(crop_name, location: str | None = None):
    """Load data for a specific crop, optionally scoped to a location.

    If a location-specific CSV exists in the form "{location}_{crop}_prices.csv",
    it will be used. Otherwise, it falls back to the global "{crop}_prices.csv".
    """
    candidates = []
    if location:
        candidates.append(f"{location}_{crop_name}_prices.csv")
    candidates.append(f"{crop_name}_prices.csv")

    for path in candidates:
        try:
            df = pd.read_csv(path, parse_dates=["month"])
            df.set_index("month", inplace=True)
            return df
        except FileNotFoundError:
            continue
    return None

# Available crops (location-agnostic names/units)
CROPS = {
    "tomato": {
        "name": "Tomato",
        "unit": "₹/kg"
    },
    "rice": {
        "name": "Rice",
        "unit": "₹/kg"
    },
    "wheat": {
        "name": "Wheat",
        "unit": "₹/kg"
    },
    "potato": {
        "name": "Potato",
        "unit": "₹/kg"
    },
    "onion": {
        "name": "Onion",
        "unit": "₹/kg"
    }
}

# Supported locations (keys are used in filenames/payloads)
LOCATIONS = {
    "food_bazaar": "Night City",
    "reliance_digital": "E District"
}

@app.get("/crops")
def get_crops():
    """Get list of available crops"""
    return {"crops": list(CROPS.keys())}

@app.get("/locations")
def get_locations():
    """Get list of available locations"""
    return {"locations": LOCATIONS}

@app.get("/crop/{crop_name}")
def get_crop_data(crop_name: str, location: str | None = None):
    """Get historical data and analysis for a specific crop"""
    if crop_name not in CROPS:
        return {"error": "Crop not found"}

    crop_data = load_crop_data(crop_name, location)
    if crop_data is None:
        return {"error": f"Data not available for {crop_name} at {location or 'default'}"}
    
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
        "historical_data": historical_data,
        "location": location,
        "location_label": LOCATIONS.get(location, None) if location else None
    }

@app.get("/forecast/{crop_name}")
def forecast(crop_name: str, months: int = 6, location: str | None = None):
    """Generate price forecast and recommendations for a crop"""
    if crop_name not in CROPS:
        return {"error": "Crop not found"}

    crop_data = load_crop_data(crop_name, location)
    if crop_data is None:
        return {"error": f"Data not available for {crop_name} at {location or 'default'}"}
    
    model = ARIMA(crop_data["price"], order=(1,1,1))
    model_fit = model.fit()
    forecast = model_fit.get_forecast(steps=months)
    predicted_mean = forecast.predicted_mean
    conf_int = forecast.conf_int()

    current_price = float(crop_data["price"].iloc[-1])

    # Compute overall slope of the forecast (trend) using linear regression
    try:
        x_idx = np.arange(len(predicted_mean))
        slope = float(np.polyfit(x_idx, predicted_mean.values, 1)[0])
    except Exception:
        slope = 0.0

    # Normalize slope to percent change per step vs current price
    slope_pct = (slope / max(current_price, 1.0)) * 100.0
    strong_up_threshold = 1.0  # >1% per step considered strong uptrend
    strong_down_threshold = -1.0  # <-1% per step considered strong downtrend

    # Per-month advice based on local delta (month-over-month change)
    previous_value = current_price
    forecast_data = []
    for date, price in predicted_mean.items():
        lower = float(conf_int.loc[date].iloc[0])
        upper = float(conf_int.loc[date].iloc[1])

        delta = float(price) - float(previous_value)
        if delta > 0:
            advice = "HOLD"
            # Stronger local move if relative change > 1%
            delta_pct = (delta / max(previous_value, 1.0)) * 100.0
            if delta_pct > strong_up_threshold:
                reasoning = "Strong upward move expected; consider holding for higher returns."
            else:
                reasoning = "Upward movement expected; holding can fetch better price."
        elif delta < 0:
            advice = "SELL FAST"
            reasoning = "Downward movement expected; sell quickly to avoid losses."
        else:
            advice = "MONITOR"
            reasoning = "Flat outlook; monitor market before acting."

        confidence = "Medium" if abs(delta) < 0.5 else ("High" if abs(delta) > 1.5 else "Medium")

        forecast_data.append({
            "month": date.strftime("%Y-%m"),
            "price": float(price),
            "lower": lower,
            "upper": upper,
            "advice": advice,
            "reasoning": reasoning,
            "confidence": confidence
        })
        previous_value = float(price)

    # Overall advice strictly by slope sign: positive => HOLD, negative => SELL FAST
    if slope_pct > strong_up_threshold:
        overall_advice = "HOLD - Rapid upward trend expected"
    elif slope_pct > 0:
        overall_advice = "HOLD - Upward trend expected"
    elif slope_pct <= strong_down_threshold:
        overall_advice = "SELL FAST - Steep downward trend expected"
    elif slope_pct < 0:
        overall_advice = "SELL FAST - Downward trend expected"
    else:
        overall_advice = "MONITOR - Flat trend expected"
    
    # Forecast-based 6M trend: average of forecast vs current
    try:
        forecast_trend_percentage = float(((predicted_mean.mean() - current_price) / max(current_price, 1.0)) * 100.0)
    except Exception:
        forecast_trend_percentage = 0.0

    return {
        "crop_name": CROPS[crop_name]["name"],
        "unit": CROPS[crop_name]["unit"],
        "current_price": current_price,
        "forecast_data": forecast_data,
        "overall_advice": overall_advice,
        "avg_forecast_price": float(predicted_mean.mean()),
        "forecast_trend_percentage": round(forecast_trend_percentage, 2),
        "location": location,
        "location_label": LOCATIONS.get(location, None) if location else None
    }

@app.get("/recommendations")
def recommendations(crop: str | None = None, location: str | None = None, months: int = 6):
    """Provide recommendations across locations and crops.

    - If crop is provided: returns best location to sell now (by current price).
    - Also returns top crops to grow (positive trend) at the specified location (or default dataset if location N/A).
    """
    result: dict[str, object] = {}

    # Best location to sell a specific crop
    if crop and crop in CROPS:
        location_prices: list[dict[str, object]] = []
        for loc_key, loc_label in LOCATIONS.items():
            df = load_crop_data(crop, loc_key)
            if df is not None and not df.empty:
                location_prices.append({
                    "location": loc_key,
                    "location_label": loc_label,
                    "current_price": float(df["price"].iloc[-1])
                })
        # Always include default/global as a fallback comparison
        df_default = load_crop_data(crop, None)
        if df_default is not None and not df_default.empty:
            location_prices.append({
                "location": None,
                "location_label": "Default",
                "current_price": float(df_default["price"].iloc[-1])
            })

        if location_prices:
            best = max(location_prices, key=lambda x: x["current_price"])
            result["best_location_for_crop"] = best
            result["all_location_prices"] = location_prices

    # Which crops to grow: pick those with highest positive 6M trend at given location
    grow_scores: list[dict[str, object]] = []
    for crop_key in CROPS.keys():
        df = load_crop_data(crop_key, location)
        if df is None or df.empty or len(df) < 12:
            continue
        recent_6m = df["price"].tail(6).mean()
        previous_6m = df["price"].tail(12).head(6).mean()
        if previous_6m == 0:
            continue
        trend_pct = ((recent_6m - previous_6m) / previous_6m) * 100
        grow_scores.append({
            "crop": crop_key,
            "crop_name": CROPS[crop_key]["name"],
            "trend_percentage": round(float(trend_pct), 2)
        })

    grow_scores.sort(key=lambda x: x["trend_percentage"], reverse=True)
    # Consider a "hike" if trend > 5%
    top_hike_crops = [g for g in grow_scores if g["trend_percentage"] > 5.0][:3]
    result["best_crops_to_grow"] = top_hike_crops
    result["location"] = location
    result["location_label"] = LOCATIONS.get(location, None) if location else None

    return result

@app.get("/")
def root():
    return {"message": "AgriVision API - Crop Price Prediction and Farmer Recommendations"}
