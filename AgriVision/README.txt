An AI-powered tool to help farmers and traders predict crop prices, detect oversupply risks, and decide the best time to sell.
Supports multiple crops: Tomato, Onion, Potato, Rice.

🚀 Features

✅ Multi-crop support – Select which crop's data to analyze (tomato, onion, potato, rice)
✅ Automatic forecasting – Uses ARIMA model to predict next 6 months' prices
✅ Graph plotting – Visualizes historical prices + forecast + confidence intervals
✅ Max/Min price detector – Prints highest and lowest monthly prices for quick reference
✅ Glut risk detection – Warns when forecasted price is likely to drop below last-year average
✅ Easy CSV input – Just add monthly prices to a simple CSV file


🗂 Project Structure
AI_Crop_Forecast/
├── tomato_prices.csv
├── onion_prices.csv
├── potato_prices.csv
├── rice_prices.csv
├── crop_price_forecast.py
└── README.md


📥 Installation

1)Install Python 3.10+ from python.org

2)Install required libraries (only once):
pip install pandas numpy matplotlib statsmodels


📊 Data Format

Each crop has its own CSV file.
Format:

month,price
2021-01-01,32
2021-02-01,34
2021-03-01,36
...


month → First day of each month (YYYY-MM-DD)

price → Average market price for that crop in that month


▶ How to Run

Run the script and choose a crop:

python crop_price_forecast.py

▶ How to Run

Run the script and choose a crop:

python crop_price_forecast.py


🧠 How It Works

ARIMA Model – Learns trend & seasonality from past prices

Confidence Intervals – Shows possible price range, not just a single value

Decision Support – Gives advice based on deviation from last-year average


🔮 Future Improvements

✅ Add auto-data fetching from government/APMC websites

✅ Add crop recommendation engine (suggest which crop to grow based on future profit)

✅ Deploy as web dashboard so farmers can check from mobile phones

DATASET LINKS:::

https://github.com/erase-d/AgriVision/blob/main/AgriVision/wheat_prices.csv
https://github.com/erase-d/AgriVision/blob/main/AgriVision/tomato_prices.csv
https://github.com/erase-d/AgriVision/blob/main/AgriVision/rice_prices.csv
https://github.com/erase-d/AgriVision/blob/main/AgriVision/potato_prices.csv
https://github.com/erase-d/AgriVision/blob/main/AgriVision/onion_prices.csv


Control Flow::::
     ┌──────────────────┐
         │  Price Data (CSV)│
         │ Tomato/Onion/etc │
         └───────┬──────────┘
                 │
                 ▼
        ┌─────────────────────┐
        │ Data Preprocessing │
        │ - Clean data       │
        │ - Handle missing   │
        │ - Convert to time  │
        └────────┬───────────┘
                 │
                 ▼
        ┌─────────────────────┐
        │  AI Forecast Model │
        │ (ARIMA/ML)        │
        └────────┬───────────┘
                 │
                 ▼
        ┌─────────────────────┐
        │  Price Analysis    │
        │ - Max/Min prices   │
        │ - Last-year avg    │
        │ - Glut risk check  │
        └────────┬───────────┘
                 │
                 ▼
        ┌─────────────────────┐
        │  Visualization     │
        │ - Price graph      │
        │ - Forecast curve   │
        │ - Confidence band  │
        └────────┬───────────┘
                 │
                 ▼
        ┌─────────────────────┐
        │ Recommendations    │
        │ - When to sell     │
        │ - Risk warning     │
        └─────────────────────┘


