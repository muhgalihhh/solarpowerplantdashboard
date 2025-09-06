# solar_dashboard_streamlit.py
# Streamlit dashboard for "Solar Power Plant" dataset (No-Upload, Responsive Layout, In-App Actual vs Prediction)
# Jalankan: streamlit run solar_dashboard_streamlit.py

import glob
import io
import math
import os
import re
from datetime import timedelta

import numpy as np
import pandas as pd
import streamlit as st

# Opsional: scikit-learn untuk regresi (fallback ke numpy jika tidak ada)
try:
    from sklearn.linear_model import LinearRegression
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

# ================= Page Config =================
st.set_page_config(
    page_title="Solar Power Plant Dashboard",
    page_icon="🔆",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("🔆 Solar Power Plant — Interactive Dashboard")
st.caption("Membaca otomatis file Excel di direktori kerja. Layout responsif, rapih, dan mudah dibaca — dengan Actual vs Prediction langsung dari model di dashboard.")

# ================= File Discovery =================
PREFERRED_FILES = [
    "Solar Power Plant Data XLSX Analyst.xlsx",
    "Solar Power Plant Dataset XLSX.xlsx",
    "Solar Power Plant Data XLSX Analyst Result.xlsx",
]

found_files = [f for f in PREFERRED_FILES if os.path.exists(f)]
if not found_files:
    patterns = [
        "*Solar*Power*Plant*Data*XLSX*Analyst*.xlsx",
        "*Solar*Power*Plant*Dataset*XLSX*.xlsx",
        "*Solar*Power*Plant*Analyst*Result*.xlsx",
    ]
    for p in patterns:
        found_files.extend(glob.glob(p))
    found_files = sorted(set(found_files))

if not found_files:
    st.error(
        "Tidak menemukan file Excel. Pastikan file ada di folder yang sama dengan script ini:\n"
        + " / ".join(PREFERRED_FILES)
    )
    st.stop()

# ================= Utilities =================
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
    """Konversi angka dengan koma desimal -> float (robust)."""
    if series.dtype == 'O':
        s = series.astype(str).str.replace(r"[^\d,\.\-]+", "", regex=True)
        both = s.str.contains(",") & s.str.contains("\.")
        s = np.where(both, s.str.replace(",", "", regex=False), s.str.replace(",", ".", regex=False))
        series = pd.Series(s)
    return pd.to_numeric(series, errors="coerce")

def parse_datetime_series(s: pd.Series) -> pd.Series:
    """Dukung format '01.01.2017-00:00' dan '2017-01-01 00:00:00'."""
    def parse_one(x):
        if pd.isna(x): return pd.NaT
        x = str(x)
        for fmt in ["%d.%m.%Y-%H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%m/%d/%Y %H:%M"]:
            try:
                return pd.to_datetime(x, format=fmt)
            except Exception:
                pass
        return pd.to_datetime(x, errors="coerce")
    return s.apply(parse_one)

def load_excel_sheet(path: str, sheet: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet)
    df.columns = [str(c).strip() for c in df.columns]

    # Cari kolom waktu utama
    date_col = coalesce_cols(df, "date_hour")
    if date_col is None:
        # Sheet turunan mungkin pakai label berbeda
        date_like = [c for c in df.columns if re.search(r"(date|time|hour)", c, re.I)]
        date_col = date_like[0] if date_like else None
    if date_col is None:
        st.error(f"Sheet '{sheet}': kolom tanggal/jam tidak ditemukan.")
        return pd.DataFrame()

    # Konversi numerik
    for k in ["wind_speed","sunshine","air_pressure","radiation","air_temperature","relative_air_humidity","system_production"]:
        col = coalesce_cols(df, k)
        if col and df[col].dtype == "O":
            df[col] = _to_float(df[col])

    # Parse datetime & urutkan
    df["timestamp"] = parse_datetime_series(df[date_col])
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # Standarisasi nama kolom inti
    rename_map = {}
    for key in STANDARD_COLS:
        col = coalesce_cols(df, key)
        if col: rename_map[col] = key
    df = df.rename(columns=rename_map)

    # Tambahkan fitur waktu + label Jam
    df["hour"] = df["timestamp"].dt.hour
    df["Jam"] = df["hour"].apply(lambda h: f"{h:02d}:00")
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24.0)
    df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24.0)
    return df

def aggregate(df, freq="H"):
    if df.empty: return df
    if freq == "D":
        g = df.set_index("timestamp").resample("D").mean(numeric_only=True).reset_index()
    elif freq == "W":
        g = df.set_index("timestamp").resample("W").mean(numeric_only=True).reset_index()
    else:
        g = df.copy()  # hourly
    return g

# ======== Models return future forecast + holdout predictions ========
def seasonal_hourly_baseline(df: pd.DataFrame, horizon: int = 24):
    """
    Baseline musiman per jam. Return:
      fcst_future: DataFrame [timestamp, yhat]
      mae: float
      holdout_df: DataFrame [timestamp, Actual, Prediction] pada area uji (15% terakhir)
    """
    if df.empty or "system_production" not in df.columns:
        return pd.DataFrame(columns=["timestamp","yhat"]), np.nan, pd.DataFrame(columns=["timestamp","Actual","Prediction"])

    cutoff = int(len(df)*0.85)
    train = df.iloc[:cutoff]
    test = df.iloc[cutoff:]
    by_hour = train.groupby(train["timestamp"].dt.hour)["system_production"].mean()

    # Prediksi holdout (validation)
    y_true = test["system_production"].values
    y_pred = test["timestamp"].apply(lambda t: by_hour.get(t.hour, train["system_production"].mean())).values
    holdout_df = pd.DataFrame({
        "timestamp": test["timestamp"].values,
        "Actual": y_true,
        "Prediction": y_pred
    })
    mae = float(np.nanmean(np.abs(y_true - y_pred))) if len(y_true)>0 else np.nan

    # Future forecast
    last_ts = df["timestamp"].max()
    future_index = pd.date_range(last_ts + pd.Timedelta(hours=1), periods=horizon, freq="H")
    forecast_values = [by_hour.get(ts.hour, train["system_production"].mean()) for ts in future_index]
    fcst_future = pd.DataFrame({"timestamp": future_index, "yhat": forecast_values})

    return fcst_future, mae, holdout_df

def regression_lag_model(df: pd.DataFrame, horizon: int = 24):
    """
    Regresi sederhana dengan lag + fitur waktu. Return:
      fcst_future: DataFrame [timestamp, yhat]
      mae: float
      holdout_df: DataFrame [timestamp, Actual, Prediction] pada area uji (15% terakhir)
    """
    if df.empty or "system_production" not in df.columns:
        return pd.DataFrame(columns=["timestamp","yhat"]), np.nan, pd.DataFrame(columns=["timestamp","Actual","Prediction"])

    work = df.copy()
    for L in [1,2,3,6,12,24]:
        work[f"lag_{L}"] = work["system_production"].shift(L)
        if "radiation" in work.columns:
            work[f"rad_lag_{L}"] = work["radiation"].shift(L)

    feature_cols = [c for c in work.columns if c.startswith("lag_") or c.startswith("rad_lag_")] + \
                   [c for c in ["sunshine","radiation","air_temperature","hour_sin","hour_cos","dayofweek"] if c in work.columns]
    work = work.dropna(subset=feature_cols + ["system_production"]).reset_index(drop=True)
    if work.empty:
        return pd.DataFrame(columns=["timestamp","yhat"]), np.nan, pd.DataFrame(columns=["timestamp","Actual","Prediction"])

    cutoff = int(len(work)*0.85)
    X_train, y_train = work.loc[:cutoff-1, feature_cols], work.loc[:cutoff-1, "system_production"]
    X_test, y_test = work.loc[cutoff:, feature_cols], work.loc[cutoff:, "system_production"]
    ts_test = work.loc[cutoff:, "timestamp"]

    if SKLEARN_OK:
        model = LinearRegression().fit(X_train, y_train)
        y_pred_test = model.predict(X_test) if len(X_test)>0 else np.array([])
        mae = float(np.nanmean(np.abs(y_test - y_pred_test))) if len(y_pred_test)>0 else np.nan
    else:
        X_ = np.c_[np.ones(len(X_train)), X_train.values]
        beta = np.linalg.lstsq(X_, y_train.values, rcond=None)[0]
        if len(X_test)>0:
            Xte_ = np.c_[np.ones(len(X_test)), X_test.values]
            y_pred_test = Xte_.dot(beta)
            mae = float(np.nanmean(np.abs(y_test.values - y_pred_test)))
        else:
            y_pred_test = np.array([])
            mae = np.nan
        class _M: pass
        model = _M(); model.beta = beta

    holdout_df = pd.DataFrame({"timestamp": ts_test.values, "Actual": y_test.values, "Prediction": y_pred_test})

    # Future forecast rekursif
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
            if ex in hist.columns:
                feat_vals[ex] = float(hist[ex].tail(24).mean())

        feat_vals["hour_sin"] = math.sin(2*math.pi*next_ts.hour/24.0)
        feat_vals["hour_cos"] = math.cos(2*math.pi*next_ts.hour/24.0)
        feat_vals["dayofweek"] = next_ts.weekday()

        fv = pd.DataFrame([{k: feat_vals.get(k, np.nan) for k in feature_cols}])
        if SKLEARN_OK:
            yhat = float(model.predict(fv)[0])
        else:
            Xfv = np.c_[np.ones(len(fv)), fv.values]
            yhat = float(Xfv.dot(model.beta)[0])

        yhat = max(0.0, yhat)
        fcst_rows.append({"timestamp": next_ts, "yhat": yhat})

        add_row = {"timestamp": next_ts, "system_production": yhat}
        for ex in ["sunshine","radiation","air_temperature"]:
            if ex in hist.columns: add_row[ex] = feat_vals.get(ex, np.nan)
        hist = pd.concat([hist, pd.DataFrame([add_row])], ignore_index=True)

    fcst_future = pd.DataFrame(fcst_rows)
    return fcst_future, mae, holdout_df

# ================= Responsive helpers =================
def two_col_or_stack(renderers):
    """
    Jika jumlah renderer <=2 → tampilkan dua kolom; jika >2 → tampil berpasangan; sisanya ke bawah.
    """
    n = len(renderers)
    if n == 0: return
    if n == 1:
        renderers[0]()
        return
    for i in range(0, n, 2):
        if i+1 < n:
            c1, c2 = st.columns(2)
            with c1: renderers[i]()
            with c2: renderers[i+1]()
        else:
            renderers[i]()

# ================= Sidebar Controls =================
st.sidebar.header("⚙️ Data & Filter")

file_choice = st.sidebar.selectbox("Pilih file Excel:", options=found_files)

xls = pd.ExcelFile(file_choice)
preferred_order = ["Solar Power Plant Data", "Feature Engineering", "Forecasting"]
sheet_names = xls.sheet_names
sheet_names = sorted(sheet_names, key=lambda s: (preferred_order.index(s) if s in preferred_order else 99, s))
sheet_choice = st.sidebar.selectbox("Pilih sheet:", options=sheet_names)

df = load_excel_sheet(file_choice, sheet_choice)
if df.empty:
    st.error("Sheet tidak memuat data yang dapat diproses.")
    st.stop()

# Filter tanggal
min_dt, max_dt = df["timestamp"].min(), df["timestamp"].max()
start_dt, end_dt = st.sidebar.slider(
    "Rentang tanggal",
    min_value=min_dt.to_pydatetime(), max_value=max_dt.to_pydatetime(),
    value=(min_dt.to_pydatetime(), max_dt.to_pydatetime()),
    format="YYYY-MM-DD"
)
mask = (df["timestamp"] >= pd.to_datetime(start_dt)) & (df["timestamp"] <= pd.to_datetime(end_dt))
dff = df.loc[mask].copy()

# Agregasi
agg = st.sidebar.radio("Agregasi", options=["Hourly","Daily","Weekly"], index=0, horizontal=True)
freq = {"Hourly":"H","Daily":"D","Weekly":"W"}[agg]
dff_agg = aggregate(dff, freq=freq)

# Smoothing
win = st.sidebar.slider("Smoothing window (jam/hari)", min_value=1, max_value=48, value=6)
if "system_production" in dff_agg.columns:
    dff_agg["system_production_smooth"] = dff_agg["system_production"].rolling(win, min_periods=1).mean()

# ================= Tabs =================
tab_overview, tab_trends, tab_rel, tab_fcst, tab_export = st.tabs(
    ["Overview", "Trends", "Relations", "Forecast (POC)", "Export"]
)

# ---------- Overview ----------
with tab_overview:
    k1,k2,k3,k4 = st.columns(4)
    with k1: st.metric("Periode", f"{start_dt.date()} → {end_dt.date()}")
    with k2: st.metric("Total Produksi (sum)", f"{float(np.nansum(dff.get('system_production', np.nan))):,.0f}")
    with k3: st.metric("Puncak Produksi (max)", f"{float(np.nanmax(dff.get('system_production', np.nan))):,.0f}")
    with k4:
        avg_rad = float(np.nanmean(dff.get("radiation"))) if "radiation" in dff.columns else float("nan")
        st.metric("Rata-rata Radiasi", f"{avg_rad:,.2f}" if not math.isnan(avg_rad) else "—")

    st.markdown("**Keterangan Jam (HH:00) & Pola Harian**")
    renderers = []

    def r_chart_hour_bar():
        if "system_production" in dff.columns:
            hour_avg = dff.groupby("Jam", sort=False)["system_production"].mean().reindex([f"{h:02d}:00" for h in range(24)])
            st.bar_chart(hour_avg.rename("Rata-rata Produksi per Jam"))
    renderers.append(r_chart_hour_bar)

    def r_table_hour_avg():
        if "system_production" in dff.columns:
            hour_avg = dff.groupby("Jam", sort=False)["system_production"].mean().reindex([f"{h:02d}:00" for h in range(24)])
            st.dataframe(hour_avg.reset_index().rename(columns={"index":"Jam","system_production":"Avg Production"}))
    renderers.append(r_table_hour_avg)

    two_col_or_stack(renderers)

# ---------- Trends ----------
with tab_trends:
    st.subheader("📈 Tren Waktu")
    sel_vars = []
    if "system_production" in dff_agg.columns: sel_vars.append("system_production")
    for optional in ["radiation","sunshine","air_temperature","wind_speed","relative_air_humidity","air_pressure"]:
        if optional in dff_agg.columns:
            sel_vars.append(optional)
    defaults = [v for v in ["system_production","radiation","sunshine"] if v in sel_vars]
    vars_show = st.multiselect("Pilih variabel:", sel_vars, default=defaults or sel_vars[:1])

    renderers = []

    def r_line_all():
        st.line_chart(dff_agg.set_index("timestamp")[vars_show])
        if "system_production_smooth" in dff_agg.columns and "system_production" in vars_show:
            st.caption(f"Garis halus (rolling mean) untuk System Production, window={win}.")
    renderers.append(r_line_all)

    def r_daily_peak():
        if "system_production" in dff.columns:
            daily = dff.set_index("timestamp")["system_production"].resample("D").agg(["sum","max","mean"]).rename(columns={"sum":"Sum","max":"Max","mean":"Mean"})
            st.line_chart(daily)
    renderers.append(r_daily_peak)

    two_col_or_stack(renderers)

# ---------- Relations ----------
with tab_rel:
    st.subheader("🔗 Korelasi & Hubungan Variabel")
    num_df = dff_agg.select_dtypes(include=["number"]).copy()
    drop_aux = [c for c in ["hour","dayofweek","month"] if c in num_df.columns]
    num_df = num_df.drop(columns=drop_aux, errors="ignore")

    renderers = []

    def r_corr():
        if num_df.shape[1] >= 2:
            corr = num_df.corr(numeric_only=True)
            st.write("**Matriks Korelasi (Pearson)**")
            st.dataframe(corr.style.background_gradient(cmap="RdYlBu", axis=None).format("{:.2f}"))
        else:
            st.info("Tidak cukup kolom numerik untuk korelasi.")
    renderers.append(r_corr)

    def r_scatter():
        if "system_production" in num_df.columns:
            numeric_cols = [c for c in num_df.columns if c != "system_production"]
            if numeric_cols:
                xvar = st.selectbox("X (predictor)", numeric_cols, key="scatter_x")
                st.scatter_chart(dff_agg[[xvar, "system_production"]])
    renderers.append(r_scatter)

    two_col_or_stack(renderers)

# ---------- Forecast (POC in-app with Actual vs Prediction) ----------
with tab_fcst:
    st.subheader("🔮 Prediksi Produksi (POC dalam App) — Actual vs Prediction")

    horizon = st.slider("Horizon prediksi (jam ke depan)", min_value=6, max_value=168, value=24, step=6)
    model_type = st.radio("Model", ["Seasonal hourly baseline", "Regresi lag + fitur waktu"], horizontal=True)

    base_cols = ["timestamp","system_production","radiation","sunshine","air_temperature","hour_sin","hour_cos","dayofweek"]
    base_cols = [c for c in base_cols if c in df.columns]
    base_df = df[base_cols].copy()

    # Jalankan model → dapatkan future forecast + holdout (actual vs prediction)
    if model_type == "Seasonal hourly baseline":
        fcst, mae, holdout = seasonal_hourly_baseline(base_df, horizon=horizon)
    else:
        fcst, mae, holdout = regression_lag_model(base_df, horizon=horizon)

    # Metrics
    # MAPE pada holdout:
    if not holdout.empty:
        err_abs = np.abs(holdout["Actual"] - holdout["Prediction"])
        with np.errstate(divide='ignore', invalid='ignore'):
            mape = np.nanmean(err_abs / np.where(holdout["Actual"]==0, np.nan, holdout["Actual"])) * 100.0
    else:
        mape = np.nan

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("MAE (Hold-out)", f"{mae:,.2f}" if not math.isnan(mae) else "—")
    with c2: st.metric("MAPE % (Hold-out)", f"{mape:,.2f}" if not math.isnan(mape) else "—")
    with c3: st.metric("N (Hold-out)", f"{len(holdout):,}")

    # Visual: kiri = Actual vs Prediction (holdout), kanan = Hist+Forecast
    def r_holdout_chart():
        if not holdout.empty:
            st.markdown("**Validation — Actual vs Prediction (15% terakhir)**")
            st.line_chart(holdout.set_index("timestamp")[["Actual","Prediction"]])
        else:
            st.info("Tidak ada data hold-out yang valid untuk divisualisasikan.")

    def r_future_chart():
      st.markdown("**Future Forecast — Histori 7 hari terakhir + Prediksi ke depan**")
      if "system_production" in dff.columns:
          hist_tail = dff.tail(24*7).copy()
          hist_tail = hist_tail[["timestamp","system_production"]].rename(columns={"system_production":"y"})
          fcst_plot = fcst.rename(columns={"yhat":"y"})
          fcst_plot["Jam"] = fcst_plot["timestamp"].dt.strftime("%H:%M")  # Tambah kolom jam
          plot_df = pd.concat([hist_tail, fcst_plot], ignore_index=True).set_index("timestamp")
          st.line_chart(plot_df)
          # Tambahkan tabel hasil forecast
          st.dataframe(fcst_plot.rename(columns={"y":"Forecast"}))
      else:
          st.info("Kolom produksi tidak tersedia untuk plot histori + forecast.")


# ---------- Export ----------
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
st.caption("Actual vs Prediction diturunkan langsung dari model di dalam dashboard (hold-out 15%). Layout otomatis dua kolom jika lebar, dan stack ke bawah bila konten banyak.")
