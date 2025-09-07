import glob
import io
import math
import os
import re
from datetime import timedelta

import numpy as np
import pandas as pd
import streamlit as st

# Optional libs
try:
    import altair as alt
    ALTAIR_OK = True
except Exception:
    ALTAIR_OK = False

PLOTLY_IMPORT_ERROR = None
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_OK = True
except Exception as _e:
    PLOTLY_OK = False
    PLOTLY_IMPORT_ERROR = _e

try:
    import warnings

    from sklearn.ensemble import (GradientBoostingRegressor,
                                  RandomForestRegressor)
    from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
    from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                                 r2_score)
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor
    warnings.filterwarnings('ignore')
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

# ---------- Page Config ----------
st.set_page_config(page_title="Solar Power Plant Dashboard ", page_icon="🔆", layout="wide", initial_sidebar_state="expanded")
st.title("🔆 Solar Power Plant — Dashboard")
st.caption("Membaca otomatis file Excel yang diunggah. Lengkap: Overview, Analisis, Forecasting, dan Kasus Bisnis.")

# Early diagnostic warning if Plotly gagal di-import
if not 'PLOTLY_OK' in globals() or not PLOTLY_OK:
    with st.sidebar.expander("⚠️ Plotly tidak aktif — klik untuk detail", expanded=True):
        st.warning("Plotly tidak tersedia, fallback ke Altair / chart bawaan. Lihat penyebab di bawah.")
        if PLOTLY_IMPORT_ERROR:
            st.code(f"{type(PLOTLY_IMPORT_ERROR).__name__}: {PLOTLY_IMPORT_ERROR}")
            st.markdown("**Solusi umum:**")
            st.markdown("""
1. Pastikan environment yang menjalankan Streamlit sama dengan environment tempat Anda meng-install dependency.
2. Upgrade / reinstall plotly & kompatibilitas numpy:
   - Windows PowerShell:
     ```powershell
     pip install --upgrade pip
     pip install --upgrade plotly numpy
     ```
3. Jika error terkait `numpy.bool` atau tipe deprecated: downgrade numpy ke versi < 2.0 atau upgrade plotly terbaru.
4. Jika memakai `trendline="ols"` (scatter), instal `statsmodels`:
     ```powershell
     pip install statsmodels
     ```
5. Restart Streamlit setelah instalasi: tutup app lalu jalankan lagi.
            """)
        else:
            st.info("Tidak ada pesan error yang tertangkap (kemungkinan variabel environment berbeda). Coba jalankan `python -c \"import plotly; print(plotly.__version__)\"` di terminal yang sama.")

# ---------- File Discovery ----------
DATASET_KEYWORD = "dataset"  # case-insensitive substring yang wajib ada

# Cari semua xlsx di folder kerja yang mengandung kata 'Dataset'
candidate_patterns = ["*.xlsx", "*Solar*Power*Plant*Dataset*.xlsx"]
all_candidates = []
for pat in candidate_patterns:
    all_candidates.extend(glob.glob(pat))

# Filter hanya yang mengandung kata 'dataset' (case-insensitive)
found_files = [f for f in sorted(set(all_candidates)) if re.search(DATASET_KEYWORD, f, re.IGNORECASE)]

if not found_files:
    st.error("Tidak menemukan file dataset (nama harus mengandung kata 'Dataset'). Letakkan file di folder ini.")
    st.stop()

# Jika lebih dari satu, tetap beri pilihan; jika hanya satu, langsung pakai tanpa selectbox

# ---------- Helpers ----------
STANDARD_COLS = {
    "date_hour": ["Date - Hour (NMT)", "Date – Hour (NMT)", "Date—Hour (NMT)", "Date Hour (NMT)", "Date_Hour_NMT", "Date/Hour", "Datetime", "DateTime"],
    "wind_speed": ["Wind Speed","Wind_Speed"],
    "sunshine": ["Sunshine"],
    "air_pressure": ["Air Pressure","Air_Pressure"],
    "radiation": ["Radiation"],
    "air_temperature": ["Air Temperature","Air_Temperature"],
    "relative_air_humidity": ["Relative Air Humidity","Relative_Air_Humidity"],
    "system_production": ["System Production","System_Production","Production","SystemProduction","Actual","y","Y"]
}

def coalesce_cols(df: pd.DataFrame, name_key: str):
    for c in STANDARD_COLS[name_key]:
        if c in df.columns:
            return c
    return None

def _to_float(series: pd.Series) -> pd.Series:
    """Enhanced numeric parsing untuk format Eropa dan international"""
    if series.dtype == 'O':
        s = series.astype(str)
        # Remove any non-numeric characters except comma, dot, and minus
        s = s.str.replace(r"[^\d,\.\-]+", "", regex=True)
        
        # Handle European format (comma as decimal separator)
        # Check if we have both comma and dot in same value
        has_both = s.str.contains(",", na=False) & s.str.contains(r"\.", na=False, regex=True)
        
        # For values with both comma and dot, assume European format (dot as thousand separator)
        # e.g., "1.234,56" -> "1234.56"
        s = np.where(
            has_both, 
            s.str.replace(r"\.", "", regex=True).str.replace(",", ".", regex=False),
            # For values with only comma, assume it's decimal separator
            s.str.replace(",", ".", regex=False)
        )
        
        series = pd.Series(s)
    
    return pd.to_numeric(series, errors="coerce")

def parse_datetime_series(s: pd.Series) -> pd.Series:
    """Enhanced datetime parsing untuk format yang beragam"""
    def parse_one(x):
        if pd.isna(x): 
            return pd.NaT
        
        # Jika sudah datetime, langsung return
        if isinstance(x, pd.Timestamp):
            return x
        
        x_str = str(x).strip()
        
        # Format patterns yang akan dicoba
        formats = [
            "%d/%m/%Y %H:%M",    # 01/01/17 12:00 (dari contoh user)
            "%m/%d/%Y %H:%M",    # 01/01/17 12:00 (US format)
            "%d.%m.%Y %H:%M",    # 01.01.2017 12:00
            "%Y-%m-%d %H:%M:%S", # 2017-01-01 12:00:00
            "%Y-%m-%d %H:%M",    # 2017-01-01 12:00
            "%d-%m-%Y %H:%M",    # 01-01-2017 12:00
            "%d.%m.%Y-%H:%M",    # Original format
        ]
        
        for fmt in formats:
            try:
                return pd.to_datetime(x_str, format=fmt)
            except (ValueError, TypeError):
                continue
        
        # Fallback ke pandas auto-parsing
        try:
            return pd.to_datetime(x_str, dayfirst=True)
        except:
            return pd.to_datetime(x_str, errors="coerce")
    
    return s.apply(parse_one)

def load_excel_sheet(path: str, sheet: str) -> pd.DataFrame:
    """Enhanced Excel loading dengan data cleaning yang lebih baik"""
    try:
        df = pd.read_excel(path, sheet_name=sheet)
        df.columns = [str(c).strip() for c in df.columns]
        
        # Remove completely empty columns (Unnamed columns with all NaN)
        df = df.loc[:, ~(df.columns.str.contains('^Unnamed') & df.isnull().all())]
        
        # Find date/time column
        date_col = coalesce_cols(df, "date_hour")
        if date_col is None:
            date_like = [c for c in df.columns if re.search(r"(date|time|hour)", c, re.I)]
            date_col = date_like[0] if date_like else None
        
        if date_col is None:
            st.error(f"Sheet '{sheet}': kolom tanggal/jam tidak ditemukan.")
            return pd.DataFrame()
        
        # Clean and convert numeric columns
        numeric_cols = ["wind_speed","sunshine","air_pressure","radiation","air_temperature","relative_air_humidity","system_production"]
        for k in numeric_cols:
            col = coalesce_cols(df, k)
            if col and col in df.columns:
                df[col] = _to_float(df[col])
        
        # Parse datetime
        df["timestamp"] = parse_datetime_series(df[date_col])
        
        # Remove rows with invalid timestamps
        before_count = len(df)
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        after_count = len(df)
        
        if before_count != after_count:
            st.info(f"Removed {before_count - after_count} rows with invalid timestamps from sheet '{sheet}'")
        
        # Rename columns to standard names
        rename_map = {}
        for key in STANDARD_COLS:
            col = coalesce_cols(df, key)
            if col: 
                rename_map[col] = key
        df = df.rename(columns=rename_map)
        
        # Create time features
        df["hour"] = df["timestamp"].dt.hour
        df["Jam"] = df["hour"].apply(lambda h: f"{h:02d}:00")
        df["dayofweek"] = df["timestamp"].dt.dayofweek
        df["month"] = df["timestamp"].dt.month
        df["day_of_year"] = df["timestamp"].dt.dayofyear
        
        # Circular encoding for time features
        df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24.0)
        df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24.0)
        df["month_sin"] = np.sin(2*np.pi*df["month"]/12.0)
        df["month_cos"] = np.cos(2*np.pi*df["month"]/12.0)
        df["day_sin"] = np.sin(2*np.pi*df["day_of_year"]/365.0)
        df["day_cos"] = np.cos(2*np.pi*df["day_of_year"]/365.0)
        
        # Add solar-specific features jika ada radiation dan production
        if "radiation" in df.columns and "system_production" in df.columns:
            # Solar efficiency ratio
            df["solar_efficiency"] = np.where(
                df["radiation"] > 0,
                df["system_production"] / df["radiation"],
                0
            )
            
            # Daytime indicator (biasanya produksi solar hanya jam 6-18)
            df["is_daytime"] = ((df["hour"] >= 6) & (df["hour"] <= 18)).astype(int)
            
            # Peak hours indicator (biasanya jam 10-15)
            df["is_peak_solar"] = ((df["hour"] >= 10) & (df["hour"] <= 15)).astype(int)
        
        # Data quality info
        total_rows = len(df)
        production_col = "system_production"
        if production_col in df.columns:
            non_zero_production = (df[production_col] > 0).sum()
            st.info(f"Sheet '{sheet}' loaded: {total_rows:,} rows, {non_zero_production:,} with production > 0 ({non_zero_production/total_rows:.1%})")
        
        return df
        
    except Exception as e:
        st.error(f"Error loading sheet '{sheet}': {str(e)}")
        return pd.DataFrame()

def aggregate(df, freq="H"):
    if df.empty: return df
    if freq == "D":
        g = df.set_index("timestamp").resample("D").mean(numeric_only=True).reset_index()
    elif freq == "W":
        g = df.set_index("timestamp").resample("W").mean(numeric_only=True).reset_index()
    else:
        g = df.copy()
    return g

# ---------- Simple Models ----------
def seasonal_hourly_baseline(df: pd.DataFrame, horizon: int = 24):
    """Improved seasonal baseline model dengan consideration untuk pola solar"""
    if df.empty or "system_production" not in df.columns:
        return pd.DataFrame(columns=["timestamp","yhat"]), np.nan, pd.DataFrame(columns=["timestamp","Actual","Prediction"])
    
    # Improved train/test split - 90% train / 10% test
    min_train_size = min(24*3, int(len(df)*0.8))  # At least 3 days or 80%
    cutoff = max(min_train_size, int(len(df)*0.90))  # 90% untuk training
    cutoff = min(cutoff, len(df) - 12)  # Ensure we have at least 12 hours for testing
    
    train = df.iloc[:cutoff]
    test = df.iloc[cutoff:]
    
    # Create sophisticated seasonal patterns
    # 1. Hourly pattern by month (seasonal adjustment)
    train_hours = train["timestamp"].dt.hour
    train_months = train["timestamp"].dt.month
    train_dow = train["timestamp"].dt.dayofweek
    
    if len(train) > 0:
        by_hour_month = train.groupby([train_hours, train_months])["system_production"].mean()
        by_hour_fallback = train.groupby(train_hours)["system_production"].mean()
        by_hour_dow = train.groupby([train_hours, train_dow])["system_production"].mean()
    else:
        by_hour_month = None
        by_hour_fallback = pd.Series(dtype=float)
        by_hour_dow = pd.Series(dtype=float)
    
    def get_prediction(timestamp):
        hour = timestamp.hour
        month = getattr(timestamp, 'month', None)
        dow = timestamp.weekday()
        
        # Try seasonal pattern first
        if by_hour_month is not None and month is not None:
            if (hour, month) in by_hour_month.index:
                return float(by_hour_month.loc[(hour, month)])
        
        # Try day-of-week pattern
        if (hour, dow) in by_hour_dow.index:
            return float(by_hour_dow.loc[(hour, dow)])
        
        # Fallback to hourly average
        if hour in by_hour_fallback.index:
            return float(by_hour_fallback.loc[hour])
        
        # Final fallback
        return float(train["system_production"].mean())
    
    # Predictions for test set
    if len(test) > 0:
        y_true = test["system_production"].values
        y_pred = test["timestamp"].apply(get_prediction).values
        mae = float(np.nanmean(np.abs(y_true - y_pred)))
        holdout_df = pd.DataFrame({
            "timestamp": test["timestamp"].values, 
            "Actual": y_true, 
            "Prediction": y_pred
        })
    else:
        mae = np.nan
        holdout_df = pd.DataFrame(columns=["timestamp","Actual","Prediction"])
    
    # Future forecasts
    last_ts = df["timestamp"].max()
    future_timestamps = pd.date_range(last_ts + pd.Timedelta(hours=1), periods=horizon, freq="H")
    forecast_values = [get_prediction(ts) for ts in future_timestamps]
    
    fcst_future = pd.DataFrame({
        "timestamp": future_timestamps, 
        "yhat": forecast_values
    })
    
    return fcst_future, mae, holdout_df

def regression_lag_model(df: pd.DataFrame, horizon: int = 24):
    """Enhanced linear regression model untuk data solar power dengan feature engineering yang lebih baik"""
    if df.empty or "system_production" not in df.columns:
        return pd.DataFrame(columns=["timestamp","yhat"]), np.nan, pd.DataFrame(columns=["timestamp","Actual","Prediction"])
    
    work = df.copy()
    
    # Enhanced lag features - focus on solar-relevant lags
    lag_periods = [1, 2, 3, 6, 12, 24, 48]  # Include 48h for day-to-day pattern
    for L in lag_periods:
        if L <= len(work):
            work[f"lag_{L}"] = work["system_production"].shift(L)
            # Add radiation lags (important for solar prediction)
            if "radiation" in work.columns:
                work[f"rad_lag_{L}"] = work["radiation"].shift(L)
            # Add temperature lags (affects solar panel efficiency)
            if "air_temperature" in work.columns:
                work[f"temp_lag_{L}"] = work["air_temperature"].shift(L)
    
    # Weather-based features
    weather_features = []
    if "radiation" in work.columns:
        weather_features.append("radiation")
        # Radiation intensity categories
        work["rad_high"] = (work["radiation"] > work["radiation"].quantile(0.75)).astype(int)
        weather_features.append("rad_high")
        
    if "sunshine" in work.columns:
        weather_features.append("sunshine")
        
    if "air_temperature" in work.columns:
        weather_features.append("air_temperature")
        # Temperature efficiency (solar panels work better in moderate temperatures)
        work["temp_optimal"] = ((work["air_temperature"] >= 15) & (work["air_temperature"] <= 25)).astype(int)
        weather_features.append("temp_optimal")
    
    # Solar-specific time features
    time_features = ["hour_sin", "hour_cos"]
    if "month_sin" in work.columns:
        time_features.extend(["month_sin", "month_cos"])
    if "day_sin" in work.columns:
        time_features.extend(["day_sin", "day_cos"])
    if "is_daytime" in work.columns:
        time_features.append("is_daytime")
    if "is_peak_solar" in work.columns:
        time_features.append("is_peak_solar")
    
    # Combine all feature types
    lag_features = [c for c in work.columns if c.startswith("lag_") or c.startswith("rad_lag_") or c.startswith("temp_lag_")]
    available_features = [c for c in weather_features + time_features if c in work.columns]
    feature_cols = lag_features + available_features
    
    # Clean data
    required_cols = feature_cols + ["system_production"]
    work = work.dropna(subset=required_cols).reset_index(drop=True)
    
    if work.empty:
        return pd.DataFrame(columns=["timestamp","yhat"]), np.nan, pd.DataFrame(columns=["timestamp","Actual","Prediction"])
    
    # Train/test split - 90% train / 10% test
    min_train_size = min(24*3, int(len(work)*0.8))  # At least 3 days
    cutoff = max(min_train_size, int(len(work)*0.90))  # 90% untuk training
    cutoff = min(cutoff, len(work) - 12)  # Ensure we have at least 12 hours for testing
    
    X_train, y_train = work.loc[:cutoff-1, feature_cols], work.loc[:cutoff-1, "system_production"]
    X_test, y_test = work.loc[cutoff:, feature_cols], work.loc[cutoff:, "system_production"]
    ts_test = work.loc[cutoff:, "timestamp"]
    
    # Model training
    if SKLEARN_OK:
        model = LinearRegression().fit(X_train, y_train)
        y_pred_test = model.predict(X_test) if len(X_test)>0 else np.array([])
        # Ensure non-negative predictions for solar production
        y_pred_test = np.maximum(y_pred_test, 0)
        mae = float(np.nanmean(np.abs(y_test - y_pred_test))) if len(y_pred_test)>0 else np.nan
    else:
        # Manual linear regression
        X_ = np.c_[np.ones(len(X_train)), X_train.values]
        try:
            beta = np.linalg.lstsq(X_, y_train.values, rcond=None)[0]
        except:
            return pd.DataFrame(columns=["timestamp","yhat"]), np.nan, pd.DataFrame(columns=["timestamp","Actual","Prediction"])
            
        if len(X_test)>0:
            Xte_ = np.c_[np.ones(len(X_test)), X_test.values]
            y_pred_test = Xte_.dot(beta)
            y_pred_test = np.maximum(y_pred_test, 0)  # Non-negative
            mae = float(np.nanmean(np.abs(y_test.values - y_pred_test)))
        else:
            y_pred_test = np.array([]); mae = np.nan
        
        class _M: pass
        model = _M(); model.beta = beta
    
    holdout_df = pd.DataFrame({
        "timestamp": ts_test.values, 
        "Actual": y_test.values, 
        "Prediction": y_pred_test
    })
    
    # Recursive future forecasting dengan improved logic
    fcst_rows = []
    hist = work.copy()
    last_ts = df["timestamp"].max()
    
    for h in range(1, horizon+1):
        next_ts = last_ts + timedelta(hours=h)
        
        # Get historical data for lag calculation
        tmp = hist.copy()
        tmp = tmp.set_index("timestamp").sort_index()
        
        def get_lag_value(col, lag_periods):
            """Get lag value with better error handling"""
            try:
                if len(tmp) >= lag_periods:
                    return float(tmp[col].iloc[-lag_periods])
                else:
                    # Use available data or mean as fallback
                    return float(tmp[col].mean()) if len(tmp) > 0 else 0.0
            except:
                return 0.0
        
        # Build feature vector
        feat_vals = {}
        
        # Lag features
        for L in lag_periods:
            if L <= len(hist):
                feat_vals[f"lag_{L}"] = get_lag_value("system_production", L)
                if "radiation" in hist.columns:
                    feat_vals[f"rad_lag_{L}"] = get_lag_value("radiation", L)
                if "air_temperature" in hist.columns:
                    feat_vals[f"temp_lag_{L}"] = get_lag_value("air_temperature", L)
        
        # Weather features (use recent averages)
        for col in ["radiation", "sunshine", "air_temperature"]:
            if col in hist.columns:
                feat_vals[col] = float(hist[col].tail(24).mean())
        
        # Derived weather features
        if "radiation" in feat_vals:
            feat_vals["rad_high"] = 1 if feat_vals["radiation"] > df["radiation"].quantile(0.75) else 0
        if "air_temperature" in feat_vals:
            feat_vals["temp_optimal"] = 1 if 15 <= feat_vals["air_temperature"] <= 25 else 0
        
        # Time features
        feat_vals["hour_sin"] = math.sin(2*math.pi*next_ts.hour/24.0)
        feat_vals["hour_cos"] = math.cos(2*math.pi*next_ts.hour/24.0)
        feat_vals["month_sin"] = math.sin(2*math.pi*next_ts.month/12.0)
        feat_vals["month_cos"] = math.cos(2*math.pi*next_ts.month/12.0)
        feat_vals["day_sin"] = math.sin(2*math.pi*next_ts.dayofyear/365.0)
        feat_vals["day_cos"] = math.cos(2*math.pi*next_ts.dayofyear/365.0)
        
        # Solar-specific features
        feat_vals["is_daytime"] = 1 if 6 <= next_ts.hour <= 18 else 0
        feat_vals["is_peak_solar"] = 1 if 10 <= next_ts.hour <= 15 else 0
        
        # Create feature vector
        fv = pd.DataFrame([{k: feat_vals.get(k, 0.0) for k in feature_cols}])
        
        # Predict
        if SKLEARN_OK:
            yhat = float(model.predict(fv)[0])
        else:
            Xfv = np.c_[np.ones(len(fv)), fv.values]
            yhat = float(Xfv.dot(model.beta)[0])
        
        # Ensure non-negative and apply solar logic
        yhat = max(0.0, yhat)
        
        # Apply solar production logic - zero production during night
        if next_ts.hour < 6 or next_ts.hour > 18:
            yhat = 0.0
        
        fcst_rows.append({"timestamp": next_ts, "yhat": yhat})
        
        # Update history for next iteration
        add_row = {"timestamp": next_ts, "system_production": yhat}
        for col in ["radiation", "sunshine", "air_temperature"]:
            if col in hist.columns: 
                add_row[col] = feat_vals.get(col, hist[col].mean())
        
        hist = pd.concat([hist, pd.DataFrame([add_row])], ignore_index=True)
    
    fcst_future = pd.DataFrame(fcst_rows)
    return fcst_future, mae, holdout_df

def random_forest_model(df: pd.DataFrame, horizon: int = 24, n_estimators: int = 100):
    """Random Forest Regressor yang disederhanakan untuk solar power prediction"""
    if df.empty or "system_production" not in df.columns or not SKLEARN_OK:
        return pd.DataFrame(columns=["timestamp","yhat"]), np.nan, pd.DataFrame(columns=["timestamp","Actual","Prediction"])
    
    work = df.copy()
    # Simplified feature engineering - hanya lag yang penting
    lag_periods = [1, 2, 3, 6, 12, 24]  # Fokus pada lag yang relevan
    
    for L in lag_periods:
        if L <= len(work):
            work[f"lag_{L}"] = work["system_production"].shift(L)
            if "radiation" in work.columns:
                work[f"rad_lag_{L}"] = work["radiation"].shift(L)
    
    # Simplified weather interactions - hanya yang paling penting
    if "radiation" in work.columns and "air_temperature" in work.columns:
        work["rad_temp_interaction"] = work["radiation"] * work["air_temperature"]
    
    # Select features
    lag_features = [c for c in work.columns if c.startswith("lag_") or c.startswith("rad_lag_")]
    weather_features = [c for c in ["radiation", "air_temperature", "sunshine", "rad_temp_interaction"] if c in work.columns]
    time_features = [c for c in ["hour_sin", "hour_cos", "is_daytime", "is_peak_solar"] if c in work.columns]
    
    feature_cols = lag_features + weather_features + time_features
    
    work = work.dropna(subset=feature_cols + ["system_production"]).reset_index(drop=True)
    if work.empty:
        return pd.DataFrame(columns=["timestamp","yhat"]), np.nan, pd.DataFrame(columns=["timestamp","Actual","Prediction"])
    
    # Train/test split - 90% train / 10% test
    min_train_size = min(24*3, int(len(work)*0.8))  # At least 3 days
    cutoff = max(min_train_size, int(len(work)*0.90))  # 90% untuk training
    cutoff = min(cutoff, len(work) - 12)  # Ensure we have at least 12 hours for testing
    
    X_train, y_train = work.loc[:cutoff-1, feature_cols], work.loc[:cutoff-1, "system_production"]
    X_test, y_test = work.loc[cutoff:, feature_cols], work.loc[cutoff:, "system_production"]
    ts_test = work.loc[cutoff:, "timestamp"]
    
    # Simplified Random Forest model
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, max_depth=10, n_jobs=-1)
    model.fit(X_train, y_train)
    
    y_pred_test = model.predict(X_test) if len(X_test)>0 else np.array([])
    # Ensure non-negative predictions
    y_pred_test = np.maximum(y_pred_test, 0)
    mae = float(np.nanmean(np.abs(y_test - y_pred_test))) if len(y_pred_test)>0 else np.nan
    
    holdout_df = pd.DataFrame({"timestamp": ts_test.values, "Actual": y_test.values, "Prediction": y_pred_test})
    
    # Simplified future forecasting
    fcst_rows = []
    hist = work.copy()
    last_ts = df["timestamp"].max()
    
    for h in range(1, horizon+1):
        next_ts = last_ts + timedelta(hours=h)
        
        # Get lag values
        def get_lag_value(col, lag_periods):
            try:
                if len(hist) >= lag_periods:
                    return float(hist[col].iloc[-lag_periods])
                else:
                    return float(hist[col].mean()) if len(hist) > 0 else 0.0
            except:
                return 0.0
        
        # Build simplified feature vector
        feat_vals = {}
        
        # Lag features
        for L in lag_periods:
            if L <= len(hist):
                feat_vals[f"lag_{L}"] = get_lag_value("system_production", L)
                if "radiation" in hist.columns:
                    feat_vals[f"rad_lag_{L}"] = get_lag_value("radiation", L)
        
        # Weather features (use recent averages)
        for col in ["radiation", "air_temperature", "sunshine"]:
            if col in hist.columns:
                feat_vals[col] = float(hist[col].tail(12).mean())  # Shorter window
        
        # Simple interaction
        if "radiation" in feat_vals and "air_temperature" in feat_vals:
            feat_vals["rad_temp_interaction"] = feat_vals["radiation"] * feat_vals["air_temperature"]
        
        # Time features
        feat_vals["hour_sin"] = math.sin(2*math.pi*next_ts.hour/24.0)
        feat_vals["hour_cos"] = math.cos(2*math.pi*next_ts.hour/24.0)
        feat_vals["is_daytime"] = 1 if 6 <= next_ts.hour <= 18 else 0
        feat_vals["is_peak_solar"] = 1 if 10 <= next_ts.hour <= 15 else 0
        
        # Create feature vector
        fv = pd.DataFrame([{k: feat_vals.get(k, 0.0) for k in feature_cols}])
        yhat = float(model.predict(fv)[0])
        
        # Apply solar production constraints
        yhat = max(0.0, yhat)
        if next_ts.hour < 6 or next_ts.hour > 18:
            yhat = 0.0
        
        fcst_rows.append({"timestamp": next_ts, "yhat": yhat})
        
        # Update history
        add_row = {"timestamp": next_ts, "system_production": yhat}
        for col in ["radiation", "air_temperature", "sunshine"]:
            if col in hist.columns: 
                add_row[col] = feat_vals.get(col, hist[col].mean())
        hist = pd.concat([hist, pd.DataFrame([add_row])], ignore_index=True)
    
    fcst_future = pd.DataFrame(fcst_rows)
    return fcst_future, mae, holdout_df

def gradient_boosting_model(df: pd.DataFrame, horizon: int = 24, n_estimators: int = 100, learning_rate: float = 0.1):
    """Gradient Boosting Regressor untuk sequential learning dan high accuracy"""
    if df.empty or "system_production" not in df.columns or not SKLEARN_OK:
        return pd.DataFrame(columns=["timestamp","yhat"]), np.nan, pd.DataFrame(columns=["timestamp","Actual","Prediction"])
    
    work = df.copy()
    # Feature engineering sama seperti Random Forest
    for L in [1,2,3,6,12,24]:
        work[f"lag_{L}"] = work["system_production"].shift(L)
        if "radiation" in work.columns:
            work[f"rad_lag_{L}"] = work["radiation"].shift(L)
    
    if "radiation" in work.columns and "air_temperature" in work.columns:
        work["rad_temp_interaction"] = work["radiation"] * work["air_temperature"]
    if "sunshine" in work.columns and "radiation" in work.columns:
        work["sun_rad_interaction"] = work["sunshine"] * work["radiation"]
    
    feature_cols = [c for c in work.columns if c.startswith("lag_") or c.startswith("rad_lag_") or c.endswith("_interaction")] + \
                   [c for c in ["sunshine","radiation","air_temperature","hour_sin","hour_cos","dayofweek"] if c in work.columns]
    
    work = work.dropna(subset=feature_cols + ["system_production"]).reset_index(drop=True)
    if work.empty:
        return pd.DataFrame(columns=["timestamp","yhat"]), np.nan, pd.DataFrame(columns=["timestamp","Actual","Prediction"])
    
    cutoff = int(len(work)*0.85)
    X_train, y_train = work.loc[:cutoff-1, feature_cols], work.loc[:cutoff-1, "system_production"]
    X_test, y_test = work.loc[cutoff:, feature_cols], work.loc[cutoff:, "system_production"]
    ts_test = work.loc[cutoff:, "timestamp"]
    
    # Model Gradient Boosting
    model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred_test = model.predict(X_test) if len(X_test)>0 else np.array([])
    mae = float(np.nanmean(np.abs(y_test - y_pred_test))) if len(y_pred_test)>0 else np.nan
    
    holdout_df = pd.DataFrame({"timestamp": ts_test.values, "Actual": y_test.values, "Prediction": y_pred_test})
    
    # Forecast future dengan logic yang sama seperti Random Forest
    fcst_rows = []
    hist = work.copy()
    last_ts = df["timestamp"].max()
    
    for h in range(1, horizon+1):
        next_ts = last_ts + timedelta(hours=h)
        tmp = hist.copy(); tmp.index = pd.DatetimeIndex(hist["timestamp"])
        
        def get_lag(col, L):
            try: return tmp.iloc[-L][col]
            except Exception: return np.nan
        
        feat_vals = {}
        for L in [1,2,3,6,12,24]:
            feat_vals[f"lag_{L}"] = get_lag("system_production", L)
            if "radiation" in hist.columns:
                feat_vals[f"rad_lag_{L}"] = get_lag("radiation", L)
        
        for ex in ["sunshine","radiation","air_temperature"]:
            if ex in hist.columns: feat_vals[ex] = float(hist[ex].tail(24).mean())
        
        feat_vals["hour_sin"] = math.sin(2*math.pi*next_ts.hour/24.0)
        feat_vals["hour_cos"] = math.cos(2*math.pi*next_ts.hour/24.0)
        feat_vals["dayofweek"] = next_ts.weekday()
        
        # Interaction features
        if "radiation" in feat_vals and "air_temperature" in feat_vals:
            feat_vals["rad_temp_interaction"] = feat_vals["radiation"] * feat_vals["air_temperature"]
        if "sunshine" in feat_vals and "radiation" in feat_vals:
            feat_vals["sun_rad_interaction"] = feat_vals["sunshine"] * feat_vals["radiation"]
        
        fv = pd.DataFrame([{k: feat_vals.get(k, np.nan) for k in feature_cols}])
        yhat = float(model.predict(fv)[0])
        yhat = max(0.0, yhat)
        
        fcst_rows.append({"timestamp": next_ts, "yhat": yhat})
        add_row = {"timestamp": next_ts, "system_production": yhat}
        for ex in ["sunshine","radiation","air_temperature"]:
            if ex in hist.columns: add_row[ex] = feat_vals.get(ex, np.nan)
        hist = pd.concat([hist, pd.DataFrame([add_row])], ignore_index=True)
    
    fcst_future = pd.DataFrame(fcst_rows)
    return fcst_future, mae, holdout_df

def polynomial_regression_model(df: pd.DataFrame, horizon: int = 24, degree: int = 2):
    """Polynomial Regression untuk menangkap non-linear relationships dengan fitur polynomial"""
    if df.empty or "system_production" not in df.columns or not SKLEARN_OK:
        return pd.DataFrame(columns=["timestamp","yhat"]), np.nan, pd.DataFrame(columns=["timestamp","Actual","Prediction"])
    
    work = df.copy()
    # Feature engineering dengan focus pada fitur yang paling relevan
    for L in [1,2,3,6,12,24]:
        work[f"lag_{L}"] = work["system_production"].shift(L)
        if "radiation" in work.columns:
            work[f"rad_lag_{L}"] = work["radiation"].shift(L)
    
    # Pilih fitur terpenting untuk polynomial expansion (untuk menghindari curse of dimensionality)
    base_feature_cols = [c for c in ["lag_1", "lag_2", "lag_3", "radiation", "air_temperature", "sunshine", "hour_sin", "hour_cos"] if c in work.columns]
    
    work = work.dropna(subset=base_feature_cols + ["system_production"]).reset_index(drop=True)
    if work.empty:
        return pd.DataFrame(columns=["timestamp","yhat"]), np.nan, pd.DataFrame(columns=["timestamp","Actual","Prediction"])
    
    cutoff = int(len(work)*0.85)
    X_train_base = work.loc[:cutoff-1, base_feature_cols]
    X_test_base = work.loc[cutoff:, base_feature_cols]
    y_train = work.loc[:cutoff-1, "system_production"]
    y_test = work.loc[cutoff:, "system_production"]
    ts_test = work.loc[cutoff:, "timestamp"]
    
    # Polynomial features
    poly = PolynomialFeatures(degree=degree, interaction_only=True, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_base)
    X_test_poly = poly.transform(X_test_base) if len(X_test_base)>0 else np.array([]).reshape(0, X_train_poly.shape[1])
    
    # Standardisasi
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_poly)
    X_test_scaled = scaler.transform(X_test_poly) if len(X_test_poly)>0 else np.array([]).reshape(0, X_train_scaled.shape[1])
    
    # Model dengan regularisasi Ridge untuk menghindari overfitting
    model = Ridge(alpha=1.0, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    y_pred_test = model.predict(X_test_scaled) if len(X_test_scaled)>0 else np.array([])
    mae = float(np.nanmean(np.abs(y_test - y_pred_test))) if len(y_pred_test)>0 else np.nan
    
    holdout_df = pd.DataFrame({"timestamp": ts_test.values, "Actual": y_test.values, "Prediction": y_pred_test})
    
    # Forecast future
    fcst_rows = []
    hist = work.copy()
    last_ts = df["timestamp"].max()
    
    for h in range(1, horizon+1):
        next_ts = last_ts + timedelta(hours=h)
        tmp = hist.copy(); tmp.index = pd.DatetimeIndex(hist["timestamp"])
        
        def get_lag(col, L):
            try: return tmp.iloc[-L][col]
            except Exception: return np.nan
        
        feat_vals = {}
        for L in [1,2,3,6,12,24]:
            if f"lag_{L}" in base_feature_cols:
                feat_vals[f"lag_{L}"] = get_lag("system_production", L)
        
        for ex in ["radiation", "air_temperature", "sunshine"]:
            if ex in base_feature_cols and ex in hist.columns: 
                feat_vals[ex] = float(hist[ex].tail(24).mean())
        
        if "hour_sin" in base_feature_cols:
            feat_vals["hour_sin"] = math.sin(2*math.pi*next_ts.hour/24.0)
        if "hour_cos" in base_feature_cols:
            feat_vals["hour_cos"] = math.cos(2*math.pi*next_ts.hour/24.0)
        
        fv_base = pd.DataFrame([{k: feat_vals.get(k, np.nan) for k in base_feature_cols}])
        fv_poly = poly.transform(fv_base)
        fv_scaled = scaler.transform(fv_poly)
        yhat = float(model.predict(fv_scaled)[0])
        yhat = max(0.0, yhat)
        
        fcst_rows.append({"timestamp": next_ts, "yhat": yhat})
        add_row = {"timestamp": next_ts, "system_production": yhat}
        for ex in ["sunshine","radiation","air_temperature"]:
            if ex in hist.columns: add_row[ex] = feat_vals.get(ex, np.nan)
        hist = pd.concat([hist, pd.DataFrame([add_row])], ignore_index=True)
    
    fcst_future = pd.DataFrame(fcst_rows)
    return fcst_future, mae, holdout_df

def two_col_or_stack(renderers):
    n = len(renderers)
    if n == 0: return
    if n == 1:
        renderers[0](); return
    for i in range(0, n, 2):
        if i+1 < n:
            c1, c2 = st.columns(2)
            with c1: renderers[i]()
            with c2: renderers[i+1]()
        else:
            renderers[i]()

# ---------- Sidebar ----------
st.sidebar.header("⚙️ Data & Filter")
if len(found_files) == 1:
    file_choice = found_files[0]
    st.sidebar.success(f"Menggunakan dataset: {file_choice}")
else:
    file_choice = st.sidebar.selectbox("Pilih file Dataset:", options=found_files)
xls = pd.ExcelFile(file_choice)
preferred_order = ["Solar Power Plant Data", "Feature Engineering", "Forecasting"]
sheet_names = sorted(xls.sheet_names, key=lambda s: (preferred_order.index(s) if s in preferred_order else 99, s))
sheet_choice = st.sidebar.selectbox("Pilih sheet:", options=sheet_names)

df = load_excel_sheet(file_choice, sheet_choice)
if df.empty:
    st.error("Sheet tidak memuat data yang dapat diproses."); st.stop()

# Filter tanggal
min_dt, max_dt = df["timestamp"].min(), df["timestamp"].max()
start_dt, end_dt = st.sidebar.slider(
    "Rentang tanggal",
    min_value=min_dt.to_pydatetime(),
    max_value=max_dt.to_pydatetime(),
    value=(min_dt.to_pydatetime(), max_dt.to_pydatetime()),
    format="YYYY-MM-DD"
)
mask = (df["timestamp"] >= pd.to_datetime(start_dt)) & (df["timestamp"] <= pd.to_datetime(end_dt))
dff = df.loc[mask].copy()

agg = st.sidebar.radio("Agregasi", options=["Hourly","Daily","Weekly"], index=0, horizontal=True)
freq = {"Hourly":"H","Daily":"D","Weekly":"W"}[agg]
dff_agg = aggregate(dff, freq=freq)

# Smoothing Options
st.sidebar.markdown("### 📊 Smoothing Options")
st.sidebar.markdown("**Smoothing windows** menghaluskan data dengan mengurangi noise dan fluktuasi acak, membantu mengidentifikasi tren yang lebih jelas.")

smoothing_method = st.sidebar.selectbox(
    "Metode Smoothing:", 
    ["Simple Moving Average", "Exponential Smoothing"],
    help="Simple Moving Average: rata-rata sederhana dari N periode. Exponential Smoothing: memberikan bobot lebih pada data terbaru."
)

win = st.sidebar.slider("Smoothing window (periode)", min_value=1, max_value=48, value=6,
                       help="Jumlah periode untuk perhitungan rata-rata bergerak. Semakin besar = lebih halus tapi kurang responsif.")

if "system_production" in dff_agg.columns:
    if smoothing_method == "Simple Moving Average":
        dff_agg["system_production_smooth"] = dff_agg["system_production"].rolling(win, min_periods=1).mean()
    else:  # Exponential Smoothing
        alpha = 2.0 / (win + 1)  # Convert window to alpha
        dff_agg["system_production_smooth"] = dff_agg["system_production"].ewm(alpha=alpha, adjust=False).mean()

# Add info about smoothing effect
if "system_production_smooth" in dff_agg.columns:
    original_std = dff_agg["system_production"].std()
    smooth_std = dff_agg["system_production_smooth"].std()
    noise_reduction = (1 - smooth_std/original_std) * 100 if original_std > 0 else 0
    st.sidebar.info(f"**Noise reduction:** {noise_reduction:.1f}%")

# ---------- Tabs ----------
tab_overview, tab_trends, tab_rel, tab_fcst, tab_biz, tab_export = st.tabs(
    ["Overview", "Trends", "Relations", "Forecast (POC)", "Business Case", "Export"]
)

# Tab Overview
with tab_overview:
    k1,k2,k3,k4 = st.columns(4)
    with k1: st.metric("Periode", f"{start_dt.date()} → {end_dt.date()}")
    with k2: st.metric("Total Produksi (sum)", f"{float(np.nansum(dff.get('system_production', np.nan))):,.0f}")
    with k3: st.metric("Puncak Produksi (max)", f"{float(np.nanmax(dff.get('system_production', np.nan))):,.0f}")
    with k4:
        avg_rad = float(np.nanmean(dff.get('radiation'))) if 'radiation' in dff.columns else float('nan')
        st.metric("Rata-rata Radiasi", f"{avg_rad:,.2f}" if not math.isnan(avg_rad) else "—")

    st.markdown("**Keterangan Jam & Pola Harian**")
    renderers = []
    def r_chart_hour_bar():
        if "system_production" in dff.columns:
            hour_avg = dff.groupby("Jam", sort=False)["system_production"].mean().reindex([f"{h:02d}:00" for h in range(24)])
            
            if PLOTLY_OK:
                # Create interactive hourly bar chart
                fig = px.bar(
                    x=hour_avg.index,
                    y=hour_avg.values,
                    title="Interactive Hourly Average Production Pattern",
                    labels={'x': 'Hour', 'y': 'Average Production'},
                    color=hour_avg.values,
                    color_continuous_scale='Oranges'
                )
                
                fig.update_traces(
                    hovertemplate=(
                        "<b>Hour: %{x}</b><br>"
                        "Avg Production: %{y:.2f}<br>"
                        "<extra></extra>"
                    )
                )
                
                fig.update_layout(
                    height=350,
                    showlegend=False,
                    coloraxis_showscale=False,
                    xaxis_title="Hour",
                    yaxis_title="Average Production"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            elif ALTAIR_OK:
                chart_df = hour_avg.reset_index().rename(columns={"index": "Jam", "system_production": "Avg_Production"})
                chart = alt.Chart(chart_df).mark_bar(
                    color="#ffb347",
                    cornerRadius=3
                ).encode(
                    x=alt.X("Jam:O", title="Jam", sort=None),
                    y=alt.Y("Avg_Production:Q", title="Rata-rata Produksi"),
                    tooltip=[
                        alt.Tooltip("Jam:O", title="Jam"),
                        alt.Tooltip("Avg_Production:Q", title="Rata-rata Produksi", format=".2f")
                    ]
                ).properties(height=300)
                st.altair_chart(chart, use_container_width=True)
            else:
                st.bar_chart(hour_avg.rename("Rata-rata Produksi per Jam"))
    renderers.append(r_chart_hour_bar)
    def r_table_hour_avg():
        if "system_production" in dff.columns:
            hour_avg = dff.groupby("Jam", sort=False)["system_production"].mean().reindex([f"{h:02d}:00" for h in range(24)])
            st.dataframe(hour_avg.reset_index().rename(columns={"index":"Jam","system_production":"Avg Production"}))
    renderers.append(r_table_hour_avg)
    two_col_or_stack(renderers)

    # Raw data preview (original loaded dataframe slice)
    with st.expander("📄 Preview Data Asli (Top 20 Rows)", expanded=False):
        st.dataframe(df.head(20), use_container_width=True)

# Tab Trends
with tab_trends:
    st.subheader("📈 Tren Waktu")
    
    # Show smoothing information if available
    if "system_production_smooth" in dff_agg.columns:
        st.info(f"ℹ️ **Smoothing aktif:** {smoothing_method} dengan window {win} periode | Noise reduction: {noise_reduction:.1f}%")
    
    # Option to show smoothed data
    show_smooth = st.checkbox("Tampilkan data yang dihaluskan (smoothed)", value=False, 
                              help=f"Menggunakan {smoothing_method.lower()} dengan window {win} periode untuk menghaluskan data dan mengurangi noise")
    
    sel_vars = []
    if "system_production" in dff_agg.columns: 
        sel_vars.append("system_production")
        if show_smooth and "system_production_smooth" in dff_agg.columns:
            sel_vars.append("system_production_smooth")
    
    for optional in ["radiation","sunshine","air_temperature","wind_speed","relative_air_humidity","air_pressure"]:
        if optional in dff_agg.columns:
            sel_vars.append(optional)
    
    defaults = [v for v in ["system_production","radiation","sunshine"] if v in sel_vars]
    if show_smooth and "system_production_smooth" in sel_vars:
        defaults = [v.replace("system_production", "system_production_smooth") if v == "system_production" else v for v in defaults]
    
    vars_show = st.multiselect("Pilih variabel:", sel_vars, default=defaults or sel_vars[:1])
    
    if vars_show:
        if PLOTLY_OK:
            # Create highly interactive Plotly chart
            fig = go.Figure()
            
            for var in vars_show:
                data_subset = dff_agg.dropna(subset=[var])
                
                # Customize line style for smoothed data
                line_width = 3 if 'smooth' in var else 2
                line_color = 'red' if 'smooth' in var else None
                var_name = var.replace('_smooth', ' (Smoothed)').replace('_', ' ').title()
                
                fig.add_trace(go.Scatter(
                    x=data_subset["timestamp"],
                    y=data_subset[var],
                    mode='lines',
                    name=var_name,
                    line=dict(width=line_width, color=line_color) if line_color else dict(width=line_width),
                    hovertemplate=(
                        f"<b>{var_name}</b><br>"
                        "Time: %{x}<br>"
                        "Value: %{y:.2f}<br>"
                        "<extra></extra>"
                    )
                ))
            
            fig.update_layout(
                title="Interactive Time Series - Hover, Zoom, and Pan Available",
                xaxis_title="Timestamp",
                yaxis_title="Value",
                height=500,
                hovermode='x unified',
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                xaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(128,128,128,0.2)'
                ),
                yaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(128,128,128,0.2)'
                )
            )
            
            # Add range slider and selector buttons for better navigation
            fig.update_layout(
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1, label="1D", step="day", stepmode="backward"),
                            dict(count=7, label="7D", step="day", stepmode="backward"),
                            dict(count=30, label="30D", step="day", stepmode="backward"),
                            dict(step="all")
                        ])
                    ),
                    rangeslider=dict(visible=True),
                    type="date"
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        elif ALTAIR_OK:
            plot_df = dff_agg.set_index("timestamp")[vars_show].reset_index()
            plot_df = plot_df.melt("timestamp", var_name="Variable", value_name="Value")
            # Remove NaN values to prevent broken lines
            plot_df = plot_df.dropna(subset=["Value"])
            
            chart = alt.Chart(plot_df).mark_line(
                strokeWidth=2.5,
                interpolate='linear'
            ).encode(
                x=alt.X("timestamp:T", title="Timestamp", axis=alt.Axis(format="%m-%d %H:%M")),
                y=alt.Y("Value:Q", title="Value"),
                color=alt.Color("Variable:N", scale=alt.Scale(scheme="tableau10")),
                tooltip=[
                    alt.Tooltip("timestamp:T", title="Timestamp", format="%Y-%m-%d %H:%M"),
                    alt.Tooltip("Variable:N", title="Variable"),
                    alt.Tooltip("Value:Q", title="Value", format=".2f")
                ]
            ).properties(height=400)
            
            # Add points for better visibility
            points = alt.Chart(plot_df).mark_circle(
                size=30,
                opacity=0.8
            ).encode(
                x="timestamp:T",
                y="Value:Q",
                color=alt.Color("Variable:N", scale=alt.Scale(scheme="tableau10")),
                tooltip=[
                    alt.Tooltip("timestamp:T", title="Timestamp", format="%Y-%m-%d %H:%M"),
                    alt.Tooltip("Variable:N", title="Variable"),
                    alt.Tooltip("Value:Q", title="Value", format=".2f")
                ]
            )
            
            combined_chart = (chart + points).resolve_scale(color='shared')
            st.altair_chart(combined_chart, use_container_width=True)
        else:
            # Clean data for streamlit fallback
            clean_df = dff_agg.set_index("timestamp")[vars_show].dropna()
            st.line_chart(clean_df)

    if "system_production" in dff.columns:
        daily = dff.set_index("timestamp")["system_production"].resample("D").agg(["sum","max","mean"]).rename(columns={"sum":"Sum","max":"Max","mean":"Mean"})
        if PLOTLY_OK:
            # Create interactive daily metrics chart
            fig = go.Figure()
            
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
            for i, metric in enumerate(['Sum', 'Max', 'Mean']):
                daily_subset = daily.dropna(subset=[metric])
                fig.add_trace(go.Scatter(
                    x=daily_subset.index,
                    y=daily_subset[metric],
                    mode='lines',
                    name=f"Daily {metric}",
                    line=dict(width=2.5, color=colors[i]),
                    hovertemplate=(
                        f"<b>Daily {metric}</b><br>"
                        "Date: %{x|%Y-%m-%d}<br>"
                        "Value: %{y:.2f}<br>"
                        "<extra></extra>"
                    )
                ))
            
            fig.update_layout(
                title="Interactive Daily Production Metrics - Sum, Max, Mean",
                xaxis_title="Date",
                yaxis_title="Production Value",
                height=400,
                hovermode='x unified',
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom", 
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                xaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(128,128,128,0.2)'
                ),
                yaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(128,128,128,0.2)'
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        elif ALTAIR_OK:
            daily_reset = daily.reset_index().melt("timestamp", var_name="Metric", value_name="Value")
            # Remove NaN values
            daily_reset = daily_reset.dropna(subset=["Value"])
            
            chart = alt.Chart(daily_reset).mark_line(
                strokeWidth=2.5,
                interpolate='linear'
            ).encode(
                x=alt.X("timestamp:T", title="Tanggal", axis=alt.Axis(format="%m-%d")),
                y=alt.Y("Value:Q", title="Nilai"),
                color=alt.Color("Metric:N", scale=alt.Scale(scheme="tableau10")),
                tooltip=[
                    alt.Tooltip("timestamp:T", title="Tanggal", format="%Y-%m-%d"),
                    alt.Tooltip("Metric:N", title="Metrik"),
                    alt.Tooltip("Value:Q", title="Nilai", format=".2f")
                ]
            ).properties(height=350)
            
            # Add points
            points = alt.Chart(daily_reset).mark_circle(
                size=40,
                opacity=0.8
            ).encode(
                x="timestamp:T",
                y="Value:Q",
                color=alt.Color("Metric:N", scale=alt.Scale(scheme="tableau10")),
                tooltip=[
                    alt.Tooltip("timestamp:T", title="Tanggal", format="%Y-%m-%d"),
                    alt.Tooltip("Metric:N", title="Metrik"),
                    alt.Tooltip("Value:Q", title="Nilai", format=".2f")
                ]
            )
            
            combined_daily = (chart + points).resolve_scale(color='shared')
            st.altair_chart(combined_daily, use_container_width=True)
        else:
            # Clean data for fallback
            clean_daily = daily.dropna()
            st.line_chart(clean_daily)

# Tab Relations
with tab_rel:
    st.subheader("🔗 Korelasi & Statistik")
    # Deskripsi variabel (singkat: asal & kegunaan)
    VAR_DESCRIPTIONS = {
        "system_production": "Output listrik aktual sistem (target utama analisis).",
        "radiation": "Intensitas radiasi matahari (driver utama produksi).",
        "sunshine": "Durasi/indikator penyinaran (proxy kondisi cerah).",
        "air_temperature": "Suhu udara; mempengaruhi efisiensi panel (terlalu panas menurunkan output).",
        "wind_speed": "Kecepatan angin; bisa membantu pendinginan panel.",
        "air_pressure": "Tekanan udara; biasanya korelasi lemah, konteks meteorologi.",
        "relative_air_humidity": "Kelembapan relatif; tinggi dapat meningkatkan difusi cahaya.",
        "solar_efficiency": "Rasio produksi terhadap radiasi (indikator performa panel).",
        "hour_sin": "Encoding siklik jam (sine) untuk model ML (diabaikan di scatter).",
        "hour_cos": "Encoding siklik jam (cosine) untuk model ML (diabaikan di scatter).",
        "month_sin": "Encoding siklik bulan (sine) untuk pola musiman (diabaikan di scatter).",
        "month_cos": "Encoding siklik bulan (cosine) untuk pola musiman (diabaikan di scatter).",
        "day_sin": "Encoding siklik hari-ke dalam setahun (sine) (diabaikan di scatter).",
        "day_cos": "Encoding siklik hari-ke dalam setahun (cosine) (diabaikan di scatter).",
        "is_daytime": "Flag 1=jam siang (6-18); bantu model filter jam produksi.",
        "is_peak_solar": "Flag 1=jam puncak (10-15); area efisiensi maksimum tipikal.",
        "lag_1": "Produksi 1 jam sebelumnya (autokorelasi).",
        "lag_2": "Produksi 2 jam sebelumnya.",
        "lag_3": "Produksi 3 jam sebelumnya.",
        "lag_6": "Produksi 6 jam sebelumnya (setengah hari).",
        "lag_12": "Produksi 12 jam sebelumnya (pola harian).",
        "lag_24": "Produksi 24 jam sebelumnya (repeat pola harian).",
        "rad_lag_1": "Radiasi 1 jam sebelumnya.",
        "rad_lag_2": "Radiasi 2 jam sebelumnya.",
        "rad_lag_3": "Radiasi 3 jam sebelumnya.",
        "rad_lag_6": "Radiasi 6 jam sebelumnya.",
        "rad_lag_12": "Radiasi 12 jam sebelumnya.",
        "rad_lag_24": "Radiasi 24 jam sebelumnya.",
        "rad_temp_interaction": "Interaksi radiasi × suhu (efek panas terhadap output).",
        "sun_rad_interaction": "Interaksi sunshine × radiasi (kondisi langit + intensitas)."
    }
    num_df = dff_agg.select_dtypes(include=["number"]).copy()
    drop_aux = [c for c in ["hour","dayofweek","month"] if c in num_df.columns]
    num_df = num_df.drop(columns=drop_aux, errors="ignore")
    if num_df.shape[1] >= 2:
        # Exclude engineered/time/binary flags from heatmap to avoid clutter & "black" grids
        heatmap_exclude = {"hour_sin","hour_cos","month_sin","month_cos","day_sin","day_cos","is_daytime","is_peak_solar","day_of_year"}
        hm_df = num_df.drop(columns=[c for c in num_df.columns if c in heatmap_exclude], errors="ignore")
        # Drop constant columns (no variance -> all NaN corr)
        constant_cols = [c for c in hm_df.columns if hm_df[c].nunique(dropna=True) <= 1]
        if constant_cols:
            hm_df = hm_df.drop(columns=constant_cols, errors="ignore")
        if hm_df.shape[1] >= 2:
            corr = hm_df.corr(numeric_only=True)
            st.markdown("**Correlation Heatmap (Pearson)** — variabel waktu/encoding & konstanta disembunyikan")
            if PLOTLY_OK:
                try:
                    import plotly.graph_objects as go  # already imported
                    hovertext = [
                        [
                            f"{row_var} vs {col_var}<br>r={corr.loc[row_var, col_var]:.2f}" + (f"<br>{VAR_DESCRIPTIONS.get(row_var,'')}" if VAR_DESCRIPTIONS.get(row_var) else "")
                            for col_var in corr.columns
                        ]
                        for row_var in corr.index
                    ]
                    fig_hm = go.Figure(data=go.Heatmap(
                        z=corr.values,
                        x=corr.columns,
                        y=corr.index,
                        colorscale='RdYlBu',
                        zmin=-1, zmax=1,
                        text=hovertext,
                        hoverinfo='text',
                        colorbar=dict(title='r')
                    ))
                    fig_hm.update_layout(height=450, margin=dict(l=60,r=20,t=40,b=60))
                    st.plotly_chart(fig_hm, use_container_width=True)
                except Exception:
                    st.dataframe(corr.style.background_gradient(cmap="RdYlBu", axis=None).format("{:.2f}"))
            else:
                st.dataframe(corr.style.background_gradient(cmap="RdYlBu", axis=None).format("{:.2f}"))

            # Correlation matrix table (same subset) with header tooltips
            st.write("**Correlation Matrix Table (subset)**")
            col_cfg = {}
            for col in corr.columns:
                desc = VAR_DESCRIPTIONS.get(col, "")
                col_cfg[col] = st.column_config.NumberColumn(col, help=desc, format="%.2f")
            st.dataframe(corr, use_container_width=True, column_config=col_cfg)
        else:
            st.info("Kolom numerik yang cukup untuk heatmap tidak tersedia setelah eksklusi.")
        if not num_df.empty:
            stats = []
            for col in num_df.columns:
                series = num_df[col].dropna()
                if series.empty:
                    continue
                stats.append({
                    "Variable": col,
                    "Mean": float(series.mean()),
                    "Median": float(series.median()),
                    "StdDev": float(series.std(ddof=1)) if len(series) > 1 else 0.0
                })
            if stats:
                stats_df = pd.DataFrame(stats).sort_values("Variable").reset_index(drop=True)
                st.markdown("**Ringkasan Statistik (Mean / Median / StdDev)** — hover nama variabel (custom HTML)")
                # Build custom HTML table with per-variable tooltip (title attr)
                def _html_escape(s):
                    return (str(s)
                            .replace('&','&amp;')
                            .replace('<','&lt;')
                            .replace('>','&gt;')
                            .replace('"','&quot;'))
                rows_html = []
                for _, r in stats_df.iterrows():
                    var = r['Variable']
                    desc = VAR_DESCRIPTIONS.get(var, "")
                    cell_var = f"<span title='{_html_escape(desc)}'>{_html_escape(var)}</span>" if desc else _html_escape(var)
                    rows_html.append(
                        f"<tr><td>{cell_var}</td><td style='text-align:right'>{r['Mean']:.2f}</td><td style='text-align:right'>{r['Median']:.2f}</td><td style='text-align:right'>{r['StdDev']:.2f}</td></tr>"
                    )
                table_html = """
                <div style='max-height:420px;overflow:auto;border:1px solid #ddd;border-radius:4px;'>
                <table style='width:100%;border-collapse:collapse;font-size:0.9rem;'>
                    <thead style='position:sticky;top:0;background:#f7f7f9;'>
                        <tr>
                            <th style='text-align:left;padding:4px 6px;'>Variable</th>
                            <th style='text-align:right;padding:4px 6px;'>Mean</th>
                            <th style='text-align:right;padding:4px 6px;'>Median</th>
                            <th style='text-align:right;padding:4px 6px;'>StdDev</th>
                        </tr>
                    </thead>
                    <tbody>
                """ + "".join(rows_html) + """
                    </tbody>
                </table>
                </div>
                <p style='font-size:0.75rem;color:#666;margin-top:4px;'>Tooltip: arahkan kursor ke nama variabel untuk deskripsi.</p>
                """
                st.markdown(table_html, unsafe_allow_html=True)
            # Tambah daftar deskripsi di expander (referensi lengkap)
            with st.expander("📘 Deskripsi Variabel Lengkap"):
                for k in sorted(set(VAR_DESCRIPTIONS.keys()) & set(num_df.columns)):
                    st.markdown(f"**{k}**: {VAR_DESCRIPTIONS[k]}")
    if "system_production" in num_df.columns:
        numeric_cols = [c for c in num_df.columns if c != "system_production"]
        # Hilangkan variabel turunan waktu & encoding supaya tidak muncul di pilihan X
        time_exclude = {"hour","dayofweek","month","day_of_year","hour_sin","hour_cos","month_sin","month_cos","day_sin","day_cos","is_daytime","is_peak_solar"}
        numeric_cols = [c for c in numeric_cols if c not in time_exclude]
        if numeric_cols:
            st.markdown("**Scatter Plot: Hubungan dengan System Production**")
            xvar = st.selectbox("X (predictor)", numeric_cols, key="scatter_x")
            scdf = dff_agg[[xvar, "system_production"]].dropna()
            # Tampilkan deskripsi variabel terpilih
            if xvar in VAR_DESCRIPTIONS:
                st.caption(f"ℹ️ {xvar}: {VAR_DESCRIPTIONS[xvar]}")
            
            if PLOTLY_OK:
                # Create interactive scatter plot; add trendline only if statsmodels tersedia
                _trendline = None
                try:
                    import statsmodels  # noqa: F401
                    _trendline = "ols"
                except Exception:
                    _trendline = None

                fig = px.scatter(
                    scdf,
                    x=xvar,
                    y="system_production",
                    title=f"Interactive Scatter: {xvar.replace('_', ' ').title()} vs System Production",
                    hover_data={
                        xvar: ':.2f',
                        'system_production': ':.2f'
                    },
                    trendline=_trendline
                )
                
                # Improve styling
                fig.update_traces(
                    marker=dict(
                        size=6,
                        opacity=0.7,
                        line=dict(width=1, color='rgba(0,0,0,0.2)')
                    ),
                    selector=dict(mode='markers')
                )
                
                # Update trendline styling if present
                if len(fig.data) > 1 and fig.data[1].mode == 'lines':
                    fig.data[1].line.color = 'red'
                    fig.data[1].line.width = 2
                
                fig.update_layout(
                    height=450,
                    xaxis_title=xvar.replace("_", " ").title(),
                    yaxis_title="System Production",
                    showlegend=False,
                    hovermode='closest'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display correlation
                corr_val = scdf[xvar].corr(scdf["system_production"])
                st.caption(f"Korelasi Pearson: {corr_val:.3f}")
                
            elif ALTAIR_OK:
                # Scatter points dengan hover
                points = alt.Chart(scdf).mark_circle(
                    size=60,
                    opacity=0.7,
                    color="#4c78a8"
                ).encode(
                    x=alt.X(xvar, title=xvar.replace("_", " ").title()),
                    y=alt.Y("system_production", title="System Production"),
                    tooltip=[
                        alt.Tooltip(xvar, title=xvar.replace("_", " ").title(), format=".2f"),
                        alt.Tooltip("system_production", title="System Production", format=".2f")
                    ]
                )
                
                # Garis tren merah
                line = alt.Chart(scdf).mark_line(
                    color="red",
                    size=2
                ).transform_regression(
                    xvar, "system_production"
                ).encode(
                    x=alt.X(xvar),
                    y=alt.Y("system_production")
                )
                
                # Gabungkan chart
                chart = (points + line).properties(height=400).resolve_scale(color='independent')
                st.altair_chart(chart, use_container_width=True)
                
                # Tampilkan korelasi
                corr_val = scdf[xvar].corr(scdf["system_production"])
                st.caption(f"Korelasi: {corr_val:.3f}")
            else:
                st.scatter_chart(dff_agg[[xvar, "system_production"]])

# Tab Forecast
with tab_fcst:
    st.subheader("🔮 Prediksi Produksi — Actual vs Prediction & Future")
    
    # Model selection dengan pilihan yang disederhanakan dan cocok untuk solar power
    col1, col2 = st.columns(2)
    with col1:
        horizon = st.slider("Horizon prediksi (jam ke depan)", min_value=6, max_value=168, value=24, step=6)
    with col2:
        model_type = st.selectbox(
            "Pilih Model ML", 
            [
                "Seasonal Hourly Baseline", 
                "Linear Regression (enhanced for solar)",
                "Random Forest (best for solar prediction)"
            ]
        )
    
    # Parameter tuning hanya untuk Random Forest
    if model_type == "Random Forest (best for solar prediction)":
        n_estimators = st.slider("Number of trees", min_value=50, max_value=150, value=100, step=25)
    
    # Prepare base data
    base_cols = ["timestamp","system_production","radiation","sunshine","air_temperature","hour_sin","hour_cos","dayofweek"]
    base_cols = [c for c in base_cols if c in df.columns]
    base_df = df[base_cols].copy()
    
    # Run selected model
    with st.spinner(f"Running {model_type}..."):
        if model_type == "Seasonal Hourly Baseline":
            fcst, mae, holdout = seasonal_hourly_baseline(base_df, horizon=horizon)
        elif model_type == "Linear Regression (enhanced for solar)":
            fcst, mae, holdout = regression_lag_model(base_df, horizon=horizon)
        elif model_type == "Random Forest (best for solar prediction)":
            fcst, mae, holdout = random_forest_model(base_df, horizon=horizon, n_estimators=n_estimators)
        else:
            fcst, mae, holdout = seasonal_hourly_baseline(base_df, horizon=horizon)
    
    # Calculate additional metrics
    if not holdout.empty:
        err_abs = np.abs(holdout["Actual"] - holdout["Prediction"])
        with np.errstate(divide='ignore', invalid='ignore'):
            mape = np.nanmean(err_abs / np.where(holdout["Actual"]==0, np.nan, holdout["Actual"])) * 100.0
        rmse = np.sqrt(np.nanmean((holdout["Actual"] - holdout["Prediction"])**2))
        
        # R² score jika sklearn tersedia
        if SKLEARN_OK:
            r2 = r2_score(holdout["Actual"], holdout["Prediction"])
        else:
            # Manual R² calculation
            ss_res = np.sum((holdout["Actual"] - holdout["Prediction"])**2)
            ss_tot = np.sum((holdout["Actual"] - np.mean(holdout["Actual"]))**2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    else:
        mape = np.nan
        rmse = np.nan
        r2 = np.nan
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1: 
        st.metric("MAE (Hold-out)", f"{mae:,.2f}" if not math.isnan(mae) else "—")
    with col2: 
        st.metric("RMSE (Hold-out)", f"{rmse:,.2f}" if not math.isnan(rmse) else "—")
    with col3: 
        st.metric("MAPE % (Hold-out)", f"{mape:,.2f}" if not math.isnan(mape) else "—")
    with col4: 
        st.metric("R² Score", f"{r2:.3f}" if not math.isnan(r2) else "—")
    
    # Model info
    st.info(f"**Model yang dipilih:** {model_type} | **Sample size:** {len(holdout):,} points | **Training split:** 90% train, 10% test")
    
    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown("**Validation — Actual vs Prediction (15% terakhir)**")
        if not holdout.empty:
            if PLOTLY_OK:
                # Create interactive validation chart
                fig = go.Figure()
                
                # Add Actual values
                fig.add_trace(go.Scatter(
                    x=holdout['timestamp'],
                    y=holdout['Actual'],
                    mode='lines',
                    name='Actual',
                    line=dict(color='blue', width=2.5),
                    hovertemplate=(
                        "<b>Actual</b><br>"
                        "Time: %{x}<br>"
                        "Value: %{y:.2f}<br>"
                        "<extra></extra>"
                    )
                ))
                
                # Add Prediction values
                fig.add_trace(go.Scatter(
                    x=holdout['timestamp'],
                    y=holdout['Prediction'],
                    mode='lines',
                    name='Prediction',
                    line=dict(color='red', width=2.5, dash='dash'),
                    hovertemplate=(
                        "<b>Prediction</b><br>"
                        "Time: %{x}<br>"
                        "Value: %{y:.2f}<br>"
                        "<extra></extra>"
                    )
                ))
                
                fig.update_layout(
                    title="Interactive Validation: Actual vs Prediction",
                    xaxis_title="Timestamp",
                    yaxis_title="Production",
                    height=400,
                    hovermode='x unified',
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            elif ALTAIR_OK:
                hdf = holdout.melt("timestamp", var_name="Series", value_name="Value")
                # Remove NaN values
                hdf = hdf.dropna(subset=["Value"])
                
                chart = alt.Chart(hdf).mark_line(
                    strokeWidth=2.5,
                    interpolate='linear'
                ).encode(
                    x=alt.X("timestamp:T", title="Timestamp", axis=alt.Axis(format="%m-%d %H:%M")),
                    y=alt.Y("Value:Q", title="Production"),
                    color=alt.Color("Series:N", scale=alt.Scale(scheme="tableau10")),
                    tooltip=[
                        alt.Tooltip("timestamp:T", title="Timestamp", format="%Y-%m-%d %H:%M"),
                        alt.Tooltip("Series:N", title="Series"),
                        alt.Tooltip("Value:Q", title="Production", format=".2f")
                    ]
                ).properties(height=350)
                
                # Add points
                points = alt.Chart(hdf).mark_circle(
                    size=35,
                    opacity=0.8
                ).encode(
                    x="timestamp:T",
                    y="Value:Q",
                    color=alt.Color("Series:N", scale=alt.Scale(scheme="tableau10")),
                    tooltip=[
                        alt.Tooltip("timestamp:T", title="Timestamp", format="%Y-%m-%d %H:%M"),
                        alt.Tooltip("Series:N", title="Series"),
                        alt.Tooltip("Value:Q", title="Production", format=".2f")
                    ]
                )
                
                combined_holdout = (chart + points).resolve_scale(color='shared')
                st.altair_chart(combined_holdout, use_container_width=True)
            else:
                clean_holdout = holdout.set_index("timestamp")[["Actual","Prediction"]].dropna()
                st.line_chart(clean_holdout)
        else:
            st.info("Tidak ada data hold-out yang valid.")
    with c_right:
        st.markdown("**Future Forecast — Histori 7 hari + Prediksi**")
        if not fcst.empty and "system_production" in dff.columns:
            hist_tail = dff.tail(24*7).copy()
            hist_tail = hist_tail[["timestamp","system_production"]].rename(columns={"system_production":"y"})
            fcp = fcst.rename(columns={"yhat":"y"})[["timestamp","y"]]
            plot_df = pd.concat([hist_tail.assign(Series="History"), fcp.assign(Series="Forecast")], ignore_index=True)
            
            if PLOTLY_OK:
                # Create interactive forecast chart
                fig = go.Figure()
                
                # Split data by series
                history_data = plot_df[plot_df['Series'] == 'History'].dropna(subset=['y'])
                forecast_data = plot_df[plot_df['Series'] == 'Forecast'].dropna(subset=['y'])
                
                # Add History trace
                fig.add_trace(go.Scatter(
                    x=history_data['timestamp'],
                    y=history_data['y'],
                    mode='lines',
                    name='History (7 days)',
                    line=dict(color='blue', width=2.5),
                    hovertemplate=(
                        "<b>History</b><br>"
                        "Time: %{x}<br>"
                        "Production: %{y:.2f}<br>"
                        "<extra></extra>"
                    )
                ))
                
                # Add Forecast trace
                fig.add_trace(go.Scatter(
                    x=forecast_data['timestamp'],
                    y=forecast_data['y'],
                    mode='lines',
                    name='Forecast',
                    line=dict(color='orange', width=2.5, dash='dot'),
                    hovertemplate=(
                        "<b>Forecast</b><br>"
                        "Time: %{x}<br>"
                        "Predicted: %{y:.2f}<br>"
                        "<extra></extra>"
                    )
                ))
                
                fig.update_layout(
                    title="Interactive Forecast: History + Future Predictions",
                    xaxis_title="Timestamp", 
                    yaxis_title="Production",
                    height=400,
                    hovermode='x unified',
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    xaxis=dict(
                        rangeselector=dict(
                            buttons=list([
                                dict(count=1, label="1D", step="day", stepmode="backward"),
                                dict(count=3, label="3D", step="day", stepmode="backward"),
                                dict(step="all")
                            ])
                        ),
                        rangeslider=dict(visible=True),
                        type="date"
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            elif ALTAIR_OK:
                # Remove NaN values
                plot_df_clean = plot_df.dropna(subset=["y"])
                
                chart = alt.Chart(plot_df_clean).mark_line(
                    strokeWidth=2.5,
                    interpolate='linear'
                ).encode(
                    x=alt.X("timestamp:T", title="Timestamp", axis=alt.Axis(format="%m-%d %H:%M")),
                    y=alt.Y("y:Q", title="Production"),
                    color=alt.Color("Series:N", scale=alt.Scale(scheme="tableau10")),
                    tooltip=[
                        alt.Tooltip("timestamp:T", title="Timestamp", format="%Y-%m-%d %H:%M"),
                        alt.Tooltip("Series:N", title="Series"),
                        alt.Tooltip("y:Q", title="Production", format=".2f")
                    ]
                ).properties(height=350)
                
                # Add points
                points = alt.Chart(plot_df_clean).mark_circle(
                    size=30,
                    opacity=0.7
                ).encode(
                    x="timestamp:T",
                    y="y:Q",
                    color=alt.Color("Series:N", scale=alt.Scale(scheme="tableau10")),
                    tooltip=[
                        alt.Tooltip("timestamp:T", title="Timestamp", format="%Y-%m-%d %H:%M"),
                        alt.Tooltip("Series:N", title="Series"),
                        alt.Tooltip("y:Q", title="Production", format=".2f")
                    ]
                )
                
                combined_forecast = (chart + points).resolve_scale(color='shared')
                st.altair_chart(combined_forecast, use_container_width=True)
            else:
                clean_plot = plot_df.set_index("timestamp").dropna()
                st.line_chart(clean_plot)
        st.dataframe(fcst.rename(columns={"yhat":"Forecast"}), use_container_width=True)
    
    # Feature importance untuk Random Forest
    if model_type == "Random Forest (best for solar prediction)" and not holdout.empty and SKLEARN_OK:
        st.markdown("---")
        st.subheader("📊 Feature Importance Analysis")
        
        # Re-run model untuk mendapatkan feature importance
        work = base_df.copy()
        lag_periods = [1, 2, 3, 6, 12, 24]
        
        for L in lag_periods:
            if L <= len(work):
                work[f"lag_{L}"] = work["system_production"].shift(L)
                if "radiation" in work.columns:
                    work[f"rad_lag_{L}"] = work["radiation"].shift(L)
        
        # Add interaction features
        if "radiation" in work.columns and "air_temperature" in work.columns:
            work["rad_temp_interaction"] = work["radiation"] * work["air_temperature"]
        
        # Select features
        lag_features = [c for c in work.columns if c.startswith("lag_") or c.startswith("rad_lag_")]
        weather_features = [c for c in ["radiation", "air_temperature", "sunshine", "rad_temp_interaction"] if c in work.columns]
        time_features = [c for c in ["hour_sin", "hour_cos", "is_daytime", "is_peak_solar"] if c in work.columns]
        
        feature_cols = lag_features + weather_features + time_features
        
        work_clean = work.dropna(subset=feature_cols + ["system_production"])
        if not work_clean.empty:
            cutoff = max(24*3, int(len(work_clean)*0.90))
            cutoff = min(cutoff, len(work_clean) - 12)
            
            X_train = work_clean.loc[:cutoff-1, feature_cols]
            y_train = work_clean.loc[:cutoff-1, "system_production"]
            
            model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, max_depth=10)
            model.fit(X_train, y_train)
            
            # Get feature importance
            importance_df = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False).head(10)
        
        # Display feature importance
        if 'importance_df' in locals() and not importance_df.empty:
            col_imp1, col_imp2 = st.columns(2)
            
            with col_imp1:
                st.markdown("**Top 10 Most Important Features**")
                
                # Create readable feature names
                importance_df['Feature_Display'] = importance_df['Feature'].replace({
                    'lag_1': '1-hour lag',
                    'lag_2': '2-hour lag', 
                    'lag_3': '3-hour lag',
                    'lag_6': '6-hour lag',
                    'lag_12': '12-hour lag',
                    'lag_24': '24-hour lag',
                    'rad_lag_1': 'Radiation 1h lag',
                    'rad_lag_2': 'Radiation 2h lag',
                    'rad_lag_3': 'Radiation 3h lag',
                    'rad_lag_6': 'Radiation 6h lag', 
                    'rad_lag_12': 'Radiation 12h lag',
                    'rad_lag_24': 'Radiation 24h lag',
                    'radiation': 'Current Radiation',
                    'air_temperature': 'Air Temperature',
                    'sunshine': 'Sunshine Hours',
                    'hour_sin': 'Hour (sine)',
                    'hour_cos': 'Hour (cosine)',
                    'dayofweek': 'Day of Week',
                    'rad_temp_interaction': 'Radiation × Temperature',
                    'sun_rad_interaction': 'Sunshine × Radiation'
                })
                
                st.dataframe(
                    importance_df[['Feature_Display', 'Importance']].rename(columns={
                        'Feature_Display': 'Feature',
                        'Importance': 'Importance Score'
                    }),
                    use_container_width=True
                )
                
            with col_imp2:
                st.markdown("**Feature Importance Visualization**")
                if PLOTLY_OK:
                    fig = go.Figure(go.Bar(
                        x=importance_df['Importance'][::-1],  # Reverse for horizontal bar
                        y=importance_df['Feature_Display'][::-1],
                        orientation='h',
                        marker_color='lightblue'
                    ))
                    fig.update_layout(
                        title="Feature Importance Scores",
                        xaxis_title="Importance Score",
                        yaxis_title="Features",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # Fallback to simple bar chart
                    chart_data = importance_df.set_index('Feature_Display')['Importance']
                    st.bar_chart(chart_data)
            
            # Insights
            st.markdown("**💡 Model Insights:**")
            top_feature = importance_df.iloc[0]['Feature_Display'] 
            top_score = importance_df.iloc[0]['Importance']
            st.write(f"• **{top_feature}** adalah fitur paling penting dengan skor {top_score:.3f}")
            
            lag_features = importance_df[importance_df['Feature'].str.contains('lag_')]['Importance'].sum()
            radiation_features = importance_df[importance_df['Feature'].str.contains('rad')]['Importance'].sum() 
            interaction_features = importance_df[importance_df['Feature'].str.contains('interaction')]['Importance'].sum()
            
            if lag_features > 0:
                st.write(f"• Fitur **lag production** berkontribusi {lag_features:.1%} dari total importance")
            if radiation_features > 0:
                st.write(f"• Fitur **radiation-related** berkontribusi {radiation_features:.1%} dari total importance") 
            if interaction_features > 0:
                st.write(f"• Fitur **interaction** berkontribusi {interaction_features:.1%} dari total importance")

# Tab Business Case
with tab_biz:
    st.subheader("🏭 Kasus Bisnis & Manfaat Dashboard")
    st.markdown("""
**Tujuan bisnis utama:**
1. **Perencanaan operasi** — memprediksi produksi per jam untuk membantu *load balancing* dan *energy dispatch*.
2. **Optimasi O&M (maintenance)** — mengidentifikasi jam puncak dan anomali (deviasi dari korelasi radiasi→produksi) sebagai sinyal potensi degradasi panel, kotoran, atau shading.
3. **Perencanaan keuangan** — agregasi harian/mingguan untuk estimasi revenue, dan perbandingan dengan target KPI.
    """)
    highlights = []
    if "system_production" in dff.columns:
        try:
            peak_row = dff.loc[dff["system_production"].idxmax()]
            highlights.append(f"- **Jam puncak** teramati sekitar **{peak_row['timestamp']}** (≈ {peak_row['system_production']:.2f}).")
        except Exception:
            pass
        if "radiation" in dff.columns:
            corr = np.corrcoef(dff["system_production"].fillna(0), dff["radiation"].fillna(0))[0,1]
            if not np.isnan(corr):
                highlights.append(f"- **Korelasi produksi–radiasi** ≈ **{corr:.2f}** (Pearson).")
    if highlights:
        st.markdown("**Sorotan cepat dari data saat ini:**\n" + "\n".join(highlights))
    st.info("Catatan: Jika tersedia kapasitas terpasang & luas panel, Anda dapat menambahkan KPI seperti **Capacity Factor** dan **Performance Ratio**.")

# Tab Export
with tab_export:
    st.subheader("⬇️ Unduh Data")
    export = dff_agg.copy()
    export = export.rename(columns={c:c.replace("_"," ").title() for c in export.columns})
    export_csv = export.to_csv(index=False).encode("utf-8")
    st.download_button("Download data teragregasi (CSV)", data=export_csv, file_name="solar_filtered_aggregated.csv", mime="text/csv")
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        export.to_excel(writer, sheet_name="Filtered Aggregated", index=False)
        try:
            if 'fcst' in globals() and not fcst.empty:
                fc = fcst.rename(columns={"yhat":"Forecast"})
                fc.to_excel(writer, sheet_name="Forecast", index=False)
        except Exception:
            pass
    st.download_button("Download Excel (Aggregated + Forecast)", data=buf.getvalue(), file_name="solar_dashboard_output.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.markdown("---")
st.caption("Dashboard: overview, analisis, forecasting, dan narasi bisnis.")
