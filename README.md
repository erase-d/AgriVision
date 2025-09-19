# AgriVision
An AI-powered tool to help farmers and traders predict crop prices, detect oversupply risks, and decide the best time to sell.
Supports multiple crops: Tomato, Onion, Potato, Rice.

🚀 Features

📊 Historical Price Data – View monthly crop prices with min, max, and average statistics.
🔮 Price Forecasting – 6-month ARIMA-based crop price prediction with confidence intervals.
🧠 Actionable Advice – Intelligent SELL / HOLD recommendations based on predicted price movements.
📍 Location-Aware Data – Supports multiple locations for price comparisons.
🌱 Best Crops to Grow – Identifies crops with strong positive trends (price hike > 5%).


.
├── backend.py                               
├── data/                                    
│   ├── food_bazaar_tomato_prices.csv        
│   ├── reliance_digital_tomato_prices.csv   
│   ├── food_bazaar_rice_prices.csv          
│   ├── reliance_digital_rice_prices.csv     
│   ├── food_bazaar_wheat_prices.csv         
│   ├── reliance_digital_wheat_prices.csv    
│   ├── food_bazaar_potato_prices.csv        
│   ├── reliance_digital_potato_prices.csv   
│   ├── food_bazaar_onion_prices.csv         
│   └── reliance_digital_onion_prices.csv    
|       
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

python backend.py

▶ How to Run

Run the script and choose a crop:

python backend.py


🧠 How It Works

ARIMA Model – Learns trend & seasonality from past prices

Confidence Intervals – Shows possible price range, not just a single value

Decision Support – Gives advice based on deviation from last-year average


🔮 Future Improvements

✅ Add auto-data fetching from government/APMC websites

✅ Add crop recommendation engine (suggest which crop to grow based on future profit)

✅ Deploy as web dashboard so farmers can check from mobile phones


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
