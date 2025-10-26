# ===========================================================
# Malawi Stock Exchange API — FastAPI application
# ===========================================================

from fastapi import FastAPI, Query, HTTPException, status  # FastAPI framework 
from typing import Optional, List  # Type hinting
from datetime import date  # Date handling
from pydantic import BaseModel  # Data validation
from sqlalchemy import create_engine, text # Database connection
from enum import Enum   # Enums for dropdowns
from dotenv import load_dotenv   # Load .env variables
import pandas as pd   # Data manipulation
import numpy as np    # Numerical operations
import os             

# ----------------------------------------------------------
# Load environment variables (.env)  and setup DB connection 
# ----------------------------------------------------------
load_dotenv()

PGHOST = os.getenv("PGHOST")
PGPORT = os.getenv("PGPORT")
PGDATABASE = os.getenv("PGDATABASE")
PGUSER = os.getenv("PGUSER")
PGPASSWORD = os.getenv("PGPASSWORD")

connection_string = f"postgresql+psycopg2://{PGUSER}:{PGPASSWORD}@{PGHOST}:{PGPORT}/{PGDATABASE}"
engine = create_engine(connection_string, pool_pre_ping=True)

# ----------------------------------------
# FastAPI initialization
# ----------------------------------------
app = FastAPI(
    title="Malawi Stock Exchange API",
    description="REST API for Malawi Stock Exchange — company and price data.",
    version="1.0"
)

# ----------------------------------------
# Helper: Sector mapping for companies
# ----------------------------------------
sector_map = {
    'AIRTEL': 'Telecommunication',
    'BHL': 'Hospitality',
    'FDHB': 'Banking',
    'FMBCH': 'Finance',
    'ICON': 'Real Estate',
    'ILLOVO': 'Agribusiness',
    'MPICO': 'Real Estate',
    'NBM': 'Banking',
    'NBS': 'Banking',
    'NICO': 'Finance',
    'NITL': 'Finance',
    'OMU': 'Finance',
    'PCL': 'Investments',
    'STANDARD': 'Banking',
    'SUNBIRD': 'Hospitality',
    'TNM': 'Telecommunication'
}

# ===========================================================
# Pydantic Models for Response Schemas
# ===========================================================
# Company Models for /companies endpoints
class Company(BaseModel):
    ticker: str
    name: str
    sector: Optional[str]
    date_listed: Optional[date]

# Detailed Company Model with total records
class CompanyDetail(Company):
    total_records: int

# Price Models for /prices endpoints
class PriceRecord(BaseModel):
    trade_date: date
    open: Optional[float]
    high: Optional[float]
    low: Optional[float]
    close: Optional[float]
    volume: Optional[float]

# Price Range Response Model with summary
class PriceRangeResponse(BaseModel):
    company: str
    summary: dict
    data: List[PriceRecord]

# Latest Price Model for /prices/latest endpoint
class LatestPriceRecord(BaseModel):
    ticker: str
    company_name: str
    sector: Optional[str]
    latest_date: date
    latest_price: Optional[float]
    previous_price: Optional[float]
    change: Optional[float]
    change_percentage: Optional[str]

# ===========================================================
# Enums for Dropdowns in Swagger UI for Validation
# ===========================================================
# 
class MonthEnum(str, Enum):
    January = "1"
    February = "2"
    March = "3"
    April = "4"
    May = "5"
    June = "6"
    July = "7"
    August = "8"
    September = "9"
    October = "10"
    November = "11"
    December = "12"

class SectorEnum(str, Enum):
    Banking = "Banking"
    Finance = "Finance"
    Real_Estate = "Real Estate"
    Hospitality = "Hospitality"
    Agribusiness = "Agribusiness"
    Telecommunication = "Telecommunication"
    Investments = "Investments"

# Function to dynamically load tickers from DB
def get_ticker_choices():
    """Dynamically load tickers from the DB for dropdowns."""
    try:
        df = pd.read_sql("SELECT ticker FROM tickers", con=engine)
        return sorted(df["ticker"].unique().tolist())
    except Exception:
        return ["NBM", "NICO", "TNM"]
# Dynamically create TickerEnum
TickerEnum = Enum("TickerEnum", {t: t for t in get_ticker_choices()})

# ===========================================================
# Utility Functions for Data Retrieval
# ===========================================================
# Get company data by ticker
def get_company_data(ticker: str):
    df = pd.read_sql("SELECT * FROM tickers", con=engine)
    df["sector"] = df["ticker"].map(sector_map)
    company = df[df["ticker"] == ticker]
    if company.empty:
        raise HTTPException(status_code=404, detail=f"Company '{ticker}' not found")
    return company

# Get price data by counter_id
def get_price_data(counter_id: str):
    df = pd.read_sql("SELECT * FROM prices_daily", con=engine)
    return df[df["counter_id"] == counter_id]

# ===========================================================
# Root route for basic info on the API homepage
# ===========================================================
# Home route with welcome message and docs link
@app.get("/")
def home():
    return {
        "message": "Welcome to the Malawi Stock Exchange API 🚀",
        "docs": "Visit /docs for interactive Swagger documentation",
        "endpoints": [
            "/companies",
            "/companies/{ticker}",
            "/prices/daily",
            "/prices/range",
            "/prices/latest"
        ]
    }

# ===========================================================
# 1️⃣ GET /companies — with optional sector filtering
# ===========================================================
@app.get("/companies", response_model=List[Company])
def get_companies(
    sector: Optional[SectorEnum] = Query(None, description="Select sector to filter companies (optional)")
):
    # Fetch companies from DB
    try:
        df = pd.read_sql("SELECT * FROM tickers", con=engine)
        df["sector"] = df["ticker"].map(sector_map)
        df = df[["ticker", "name", "sector", "date_listed"]]
        if sector:
            df = df[df["sector"].str.lower() == sector.value.lower()]
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

# ===========================================================
# 2️⃣ GET /companies/{ticker} -- detailed company info with total records
# ===========================================================
# Get detailed company info by ticker
@app.get("/companies/{ticker}", response_model=CompanyDetail)
def get_company_info(ticker: str):
    # Fetch company and count records
    try:
        company = get_company_data(ticker)
        counter_id = company["counter_id"].values[0]
        df_prices = pd.read_sql("SELECT * FROM prices_daily", con=engine)
        total_records = len(df_prices[df_prices["counter_id"] == counter_id])
        return {
            "ticker": ticker,
            "name": company["name"].values[0],
            "sector": company["sector"].values[0],
            "date_listed": company["date_listed"].values[0],
            "total_records": total_records
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

# ==============================================================================
# 3️⃣ GET /prices/daily — with date range filtering and limit option for records
# ===============================================================================
# Get daily prices with optional date range and limit
@app.get("/prices/daily", response_model=List[PriceRecord])
def get_prices_daily(
    ticker: TickerEnum = Query(..., description="Select ticker"),
    start_date: Optional[date] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[date] = Query(None, description="End date (YYYY-MM-DD)"),
    limit: int = Query(100, le=1000, description="Max records to return (default=100)")
):
    # Fetch price data with filters
    try:
        ticker = ticker.value.upper()
        company = get_company_data(ticker)
        counter_id = company["counter_id"].values[0]

        df = get_price_data(counter_id)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No price data found for {ticker}")

        # Clean and filter date
        df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
        df = df.dropna(subset=["trade_date"])

        if start_date:
            df = df[df["trade_date"] >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df["trade_date"] <= pd.Timestamp(end_date)]
        if df.empty:
            raise HTTPException(status_code=404, detail="No data in the given date range")

        # Rename price columns safely
        rename_map = {
            "open_mwk": "open",
            "high_mwk": "high",
            "low_mwk": "low",
            "close_mwk": "close"
        }
        df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

        # Keep only relevant columns
        df = df[["trade_date", "open", "high", "low", "close", "volume"]]

        # Convert to proper types for Pydantic
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

        # Convert NaN -> None and NumPy -> Python types
        records = df.sort_values("trade_date", ascending=False).head(limit).to_dict(orient="records")
        for r in records:
            for k, v in r.items():
                if pd.isna(v):
                    r[k] = None
                elif isinstance(v, (np.generic,)):
                    r[k] = v.item()

        return records

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error in /prices/daily: {str(e)}")


# ================================================================
# 4️⃣ GET /prices/range — summary statistics for a given year/month 
# ================================================================
# Get price range summary statistics for a given year/month
@app.get("/prices/range", response_model=PriceRangeResponse)
def get_prices_range(
    ticker: TickerEnum = Query(..., description="Select ticker"),
    year: int = Query(..., ge=2017, le=date.today().year, description="Select year"),
    month: Optional[MonthEnum] = Query(None, description="Optional month filter")
):
    # Fetch price data and compute summary
    try:
        ticker = ticker.value.upper()
        month_val = int(month.value) if month else None

        company = get_company_data(ticker)
        counter_id = company["counter_id"].values[0]

        df = get_price_data(counter_id)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {ticker}")

        # Clean and filter date
        df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
        df = df.dropna(subset=["trade_date"])

        df = df[df["trade_date"].dt.year == year]
        if month_val:
            df = df[df["trade_date"].dt.month == month_val]
        if df.empty:
            raise HTTPException(status_code=404, detail="No price data for the selected period")

        # Rename and cast
        rename_map = {
            "open_mwk": "open",
            "high_mwk": "high",
            "low_mwk": "low",
            "close_mwk": "close"
        }
        df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
        df = df[["trade_date", "open", "high", "low", "close", "volume"]]

        # Summary statistics
        summary = {
            "period_high": df["high"].max(skipna=True),
            "period_low": df["low"].min(skipna=True),
            "average_open": round(df["open"].mean(skipna=True), 2),
            "average_close": round(df["close"].mean(skipna=True), 2),
            "total_volume": float(df["volume"].sum(skipna=True)),
            "record_count": int(len(df))
        }

        # Replace NaN with None
        summary = {k: (None if pd.isna(v) else v) for k, v in summary.items()}

        # Convert DataFrame to JSON-safe list
        records = df.to_dict(orient="records")
        for r in records:
            for k, v in r.items():
                if pd.isna(v):
                    r[k] = None
                elif isinstance(v, (np.generic,)):
                    r[k] = v.item()

        return {
            "company": ticker,
            "summary": summary,
            "data": records
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error in /prices/range: {str(e)}")

# ======================================================================================================================
# 5️⃣ GET /prices/latest — latest price info with percentage change calculations and optional filtering for ticker/sector
# ======================================================================================================================
# Get latest price info with optional filtering
@app.get("/prices/latest", response_model=List[LatestPriceRecord])
def get_latest_prices(
    ticker: Optional[TickerEnum] = Query(None, description="Select ticker (optional)"),
    sector: Optional[SectorEnum] = Query(None, description="Filter by sector (optional)")
):
    # Fetch latest prices with optional filters
    try:
        tickers_df = pd.read_sql("SELECT * FROM tickers", con=engine)
        prices_df = pd.read_sql("SELECT * FROM prices_daily", con=engine)

        if prices_df.empty or tickers_df.empty:
            raise HTTPException(status_code=404, detail="No trading data available")

        tickers_df["sector"] = tickers_df["ticker"].map(sector_map)

        # Apply filters
        if ticker:
            ticker = ticker.value
            tickers_df = tickers_df[tickers_df["ticker"] == ticker]
            if tickers_df.empty:
                raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' not found")
        elif sector:
            tickers_df = tickers_df[tickers_df["sector"].str.lower() == sector.value.lower()]
            if tickers_df.empty:
                raise HTTPException(status_code=404, detail=f"No companies found in sector '{sector.value}'")
        
        # Compute latest prices and changes
        results = []
        for _, row in tickers_df.iterrows():
            cid = row["counter_id"]
            tk = row["ticker"]
            name = row["name"]
            sec = row["sector"]

            df = prices_df[prices_df["counter_id"] == cid]
            if df.empty:
                continue

            df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
            df = df.sort_values("trade_date", ascending=False).reset_index(drop=True)

            # Get latest and previous prices
            latest_row = df.iloc[0]
            latest_date = latest_row["trade_date"].date()
            latest_price = latest_row["close_mwk"]
            prev_price = df.iloc[1]["close_mwk"] if len(df) > 1 else None

            # Calculate change and percentage
            if prev_price and not pd.isna(prev_price):
                change = latest_price - prev_price
                change_pct = f"{round((change / prev_price) * 100, 2)}%"
            else:
                change, change_pct = None, None

            # Append result for this ticker
            results.append({
                "ticker": tk,
                "company_name": name,
                "sector": sec,
                "latest_date": latest_date,
                "latest_price": latest_price,
                "previous_price": prev_price,
                "change": change,
                "change_percentage": change_pct
            })

        if not results:
            raise HTTPException(status_code=404, detail="No recent price data found")

        return results

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
    
# ===========================================================
