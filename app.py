
import io
import sqlite3
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

APP_TITLE = "Market Command Center"
DB_PATH = Path(__file__).with_name("market_command_center.db")

SCAN_KEYS = {
    "rotation": "4% Rotation",
    "dcr": "High DCR",
    "ep": "$9M EP",
    "trend47": "4/7 Trend",
    "trend10ema": "10EMA Trend",
}

CORE_BULLISH_SCANS = {"rotation", "dcr", "trend47", "ep_positive"}
CONFIRMATION_SCAN = "trend10ema"


def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db():
    conn = get_conn()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            session_date TEXT PRIMARY KEY,
            notes TEXT DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS scan_rows (
            session_date TEXT NOT NULL,
            scan_key TEXT NOT NULL,
            symbol TEXT NOT NULL,
            sub_industry TEXT,
            rvol REAL,
            rs1m REAL,
            rs3m REAL,
            adr REAL,
            dollar_vol_b REAL,
            pct_change REAL,
            theme TEXT,
            PRIMARY KEY (session_date, scan_key, symbol)
        );

        CREATE INDEX IF NOT EXISTS idx_scan_rows_date ON scan_rows(session_date);
        CREATE INDEX IF NOT EXISTS idx_scan_rows_symbol ON scan_rows(symbol);
        CREATE INDEX IF NOT EXISTS idx_scan_rows_theme ON scan_rows(theme);
        """
    )
    conn.commit()
    conn.close()


def detect_scan_type(uploaded_file):
    """
    Detect which scan a CSV belongs to from the Thinkorswim watchlist metadata
    at the top of the export, with the filename used as a fallback.
    """
    raw = uploaded_file.getvalue()
    text = raw.decode("utf-8-sig", errors="ignore")
    head = "\n".join(text.splitlines()[:10]).lower()
    name = (getattr(uploaded_file, "name", "") or "").lower()
    haystack = f"{name}\n{head}"

    # Most specific rules first.
    if any(k in haystack for k in [
        "10ema", "10 ema", "above 10 ema", "over 10ema",
        "trending over 10ema", "trend over 10ema"
    ]):
        return "trend10ema"

    if any(k in haystack for k in [
        "4/7 trend", "4/7", "trend accumulation", "vol 4/7",
        "4 of 7", "4of7"
    ]):
        return "trend47"

    if any(k in haystack for k in [
        "high dcr", "sr.high dcr", "dcr/intraday"
    ]):
        return "dcr"

    if any(k in haystack for k in [
        "ep9 million", "ep 9 million", "9 million", "9million",
        "sr.ep9", "$9m", "9m ep"
    ]):
        return "ep"

    if any(k in haystack for k in [
        "rotation spotter", "sr.4%", "4%(intraday)", "4% rotation",
        "rotation"
    ]):
        return "rotation"

    return None


def auto_assign_uploads(uploaded_files):
    """
    Return (assigned, unknown, duplicates).
    assigned maps scan_key -> uploaded file.
    """
    assigned = {}
    unknown = []
    duplicates = []

    for f in uploaded_files or []:
        key = detect_scan_type(f)
        if key is None:
            unknown.append(f)
        elif key in assigned:
            duplicates.append((key, f))
        else:
            assigned[key] = f

    return assigned, unknown, duplicates

def read_tos_csv(uploaded_file):
    raw = uploaded_file.getvalue()
    text = raw.decode("utf-8-sig", errors="ignore")
    lines = text.splitlines()
    header_idx = None
    for i, line in enumerate(lines[:15]):
        if line.startswith("Symbol,"):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("Could not find a Thinkorswim 'Symbol' header row.")

    csv_text = "\n".join(lines[header_idx:])
    df = pd.read_csv(io.StringIO(csv_text))

    for col in ["Symbol", "Sub-Industry"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Accept common Thinkorswim column-name variations.
    aliases = {
        "rvol": ["jeffRvol.c", "RVOL", "RVol", "Relative Volume"],
        "rs1m": ["jeffRS1month.c", "RS1M", "RS 1M", "1M RS"],
        "rs3m": ["jeffRS3month.c", "RS3M", "RS 3M", "3M RS"],
        "adr": ["ADR", "ADR%"],
        "dollar_vol_b": ["jeff$VolBillion", "$VolBillion", "Dollar Vol B", "Dollar Volume B"],
        "pct_change": ["jy%change", "%Change", "% Change", "Pct Change", "Change %"],
    }
    rename = {}
    for target, candidates in aliases.items():
        for candidate in candidates:
            if candidate in df.columns:
                rename[candidate] = target
                break
    df = df.rename(columns=rename)
    df["Symbol"] = df["Symbol"].astype(str).str.upper().str.strip()
    df["Sub-Industry"] = df["Sub-Industry"].fillna("").astype(str).str.strip()

    for c in ["rvol", "rs1m", "rs3m", "adr", "dollar_vol_b", "pct_change"]:
        if c not in df.columns:
            df[c] = np.nan
        # Robustly handle numeric strings and optional percent signs.
        df[c] = (
            df[c]
            .astype(str)
            .str.replace("%", "", regex=False)
            .str.replace(",", "", regex=False)
            .replace({"nan": np.nan, "None": np.nan, "": np.nan})
        )
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df[df["Symbol"].str.len() > 0].copy()
    return df


def classify_theme(sub_industry):
    s = (sub_industry or "").lower()

    if any(k in s for k in [
        "biotechnology", "pharmaceutical", "life sciences", "health care equipment",
        "health care supplies", "health care services", "health care facilities",
        "health care technology", "managed health care", "healthcare", "drug retail"
    ]):
        return "Healthcare / Biotech"

    if "application software" in s or "systems software" in s:
        return "Software"

    if any(k in s for k in [
        "it consulting", "data processing", "outsourced services",
        "internet services", "infrastructure", "transaction & payment",
        "transaction and payment", "payment processing"
    ]):
        return "Tech Services / Infra"

    if any(k in s for k in [
        "semiconductor", "technology hardware", "storage & peripherals",
        "storage and peripherals", "electronic equipment", "communications equipment",
        "technology distributors", "electronic components"
    ]):
        return "Semis / AI Hardware"

    if any(k in s for k in [
        "oil & gas", "oil and gas", "energy equipment", "coal", "integrated oil"
    ]):
        return "Energy"

    if any(k in s for k in [
        "retail", "restaurants", "apparel", "footwear", "hotels", "resorts",
        "casinos", "gaming", "automobile", "automotive", "homefurnishing",
        "home furnishing", "leisure", "consumer products", "consumer staples",
        "household products", "personal care", "food products", "beverages",
        "distributors", "specialty stores"
    ]):
        return "Consumer / Retail"

    if any(k in s for k in [
        "bank", "financial", "insurance", "capital markets", "mortgage",
        "reit", "real estate", "asset management", "consumer finance"
    ]):
        return "Financial / REIT"

    if any(k in s for k in [
        "aerospace", "defense", "industrial", "machinery", "construction",
        "engineering", "transportation", "air freight", "ground transportation",
        "commercial services", "professional services", "environmental",
        "electrical equipment", "building products", "trading companies"
    ]):
        return "Industrials / Services"

    if any(k in s for k in [
        "electric utilities", "multi-utilities", "gas utilities", "water utilities",
        "independent power", "renewable electricity", "utilities"
    ]):
        return "Utilities / Power"

    if any(k in s for k in [
        "media", "entertainment", "broadcast", "telecom", "wireless",
        "interactive media", "advertising"
    ]):
        return "Media / Telecom"

    if any(k in s for k in [
        "metals", "mining", "gold", "silver", "steel", "aluminum",
        "chemicals", "paper", "forest products", "materials"
    ]):
        return "Materials / Metals"

    return "Other"


def save_session(session_date, uploads, notes=""):
    parsed = {}
    for scan_key, uploaded in uploads.items():
        if uploaded is None:
            continue
        df = read_tos_csv(uploaded)
        df["theme"] = df["Sub-Industry"].map(classify_theme)
        parsed[scan_key] = df

    if len(parsed) < 4:
        raise ValueError("Please upload at least the original four scans.")

    conn = get_conn()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO sessions(session_date, notes) VALUES (?, ?)",
            (session_date.isoformat(), notes),
        )
        conn.execute("DELETE FROM scan_rows WHERE session_date = ?", (session_date.isoformat(),))

        rows = []
        for scan_key, df in parsed.items():
            for _, r in df.iterrows():
                rows.append((
                    session_date.isoformat(),
                    scan_key,
                    r["Symbol"],
                    r["Sub-Industry"],
                    r["rvol"],
                    r["rs1m"],
                    r["rs3m"],
                    r["adr"],
                    r["dollar_vol_b"],
                    r["pct_change"],
                    r["theme"],
                ))
        conn.executemany(
            """
            INSERT OR REPLACE INTO scan_rows
            (session_date, scan_key, symbol, sub_industry, rvol, rs1m, rs3m, adr,
             dollar_vol_b, pct_change, theme)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
    finally:
        conn.close()
    return parsed


def load_rows(start=None, end=None):
    conn = get_conn()
    q = "SELECT * FROM scan_rows"
    params = []
    clauses = []
    if start:
        clauses.append("session_date >= ?")
        params.append(str(start))
    if end:
        clauses.append("session_date <= ?")
        params.append(str(end))
    if clauses:
        q += " WHERE " + " AND ".join(clauses)
    df = pd.read_sql_query(q, conn, params=params)
    conn.close()

    # Ensure loaded numeric fields stay numeric.
    for c in ["rvol", "rs1m", "rs3m", "adr", "dollar_vol_b", "pct_change"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def get_dates():
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT session_date, notes FROM sessions ORDER BY session_date DESC", conn
    )
    conn.close()
    return df


def make_core_bullish(rows):
    if rows.empty:
        return rows.copy()
    pieces = []

    for key in ["rotation", "dcr", "trend47"]:
        part = rows[rows["scan_key"] == key].copy()
        part["source"] = key
        pieces.append(part)

    ep = rows[(rows["scan_key"] == "ep") & (rows["pct_change"] > 0)].copy()
    ep["source"] = "ep_positive"
    pieces.append(ep)

    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def session_counts(rows):
    out = []
    for d, g in rows.groupby("session_date"):
        out.append({
            "date": d,
            "4% Rotation": int((g["scan_key"] == "rotation").sum()),
            "High DCR": int((g["scan_key"] == "dcr").sum()),
            "Positive $9M EP": int(((g["scan_key"] == "ep") & (g["pct_change"] > 0)).sum()),
            "Negative $9M EP": int(((g["scan_key"] == "ep") & (g["pct_change"] < 0)).sum()),
            "4/7 Trend": int((g["scan_key"] == "trend47").sum()),
            "10EMA Trend": int((g["scan_key"] == "trend10ema").sum()),
        })
    return pd.DataFrame(out).sort_values("date") if out else pd.DataFrame()


def theme_participation_for_date(rows, session_date):
    day = rows[rows["session_date"] == session_date]
    bull = make_core_bullish(day)
    if bull.empty:
        return pd.DataFrame(columns=["theme", "stocks"])
    return (
        bull.drop_duplicates(["symbol", "theme"])
        .groupby("theme")["symbol"].nunique()
        .sort_values(ascending=False)
        .rename("stocks")
        .reset_index()
    )


def theme_persistence(rows, themes=None):
    recs = []
    for d in sorted(rows["session_date"].unique()):
        x = theme_participation_for_date(rows, d)
        for _, r in x.iterrows():
            recs.append({"date": d, "theme": r["theme"], "stocks": int(r["stocks"])})
    df = pd.DataFrame(recs)
    if themes and not df.empty:
        df = df[df["theme"].isin(themes)]
    return df


def rs_quality(rows, session_date):
    day = rows[rows["session_date"] == session_date]
    bull = make_core_bullish(day)
    if bull.empty:
        return pd.DataFrame(columns=["theme", "rs1m", "rs3m", "stocks"])
    u = bull.sort_values("rs3m", ascending=False).drop_duplicates("symbol")
    return (
        u.groupby("theme")
        .agg(
            rs1m=("rs1m", "median"),
            rs3m=("rs3m", "median"),
            stocks=("symbol", "nunique"),
        )
        .reset_index()
        .sort_values(["rs3m", "rs1m"], ascending=False)
    )


def overlap_table(rows, session_date):
    day = rows[rows["session_date"] == session_date].copy()
    if day.empty:
        return pd.DataFrame()

    day["bullish_scan"] = True
    day.loc[
        (day["scan_key"] == "ep") & ~(day["pct_change"] > 0),
        "bullish_scan"
    ] = False
    bull = day[day["bullish_scan"]].copy()

    def labels(g):
        out = []
        scan_set = set(g["scan_key"])
        for key in ["rotation", "dcr", "trend47", "trend10ema"]:
            if key in scan_set:
                out.append(SCAN_KEYS[key])
        if ((g["scan_key"] == "ep") & (g["pct_change"] > 0)).any():
            out.append("EP+")
        return ", ".join(out)

    base = (
        bull.groupby("symbol")
        .agg(
            overlap=("scan_key", "nunique"),
            theme=("theme", "first"),
            sub_industry=("sub_industry", "first"),
            rvol=("rvol", "max"),
            rs1m=("rs1m", "max"),
            rs3m=("rs3m", "max"),
            pct_change=("pct_change", "max"),
        )
        .reset_index()
    )
    labs = bull.groupby("symbol").apply(labels).rename("scans").reset_index()
    base = base.merge(labs, on="symbol", how="left")

    tenema_symbols = set(day.loc[day["scan_key"] == "trend10ema", "symbol"])
    base["10ema_confirm"] = base["symbol"].isin(tenema_symbols)

    base["score"] = (
        base["overlap"] * 25
        + base["10ema_confirm"].astype(int) * 10
        + base["rs3m"].fillna(100).clip(50, 250) * 0.20
        + base["rs1m"].fillna(100).clip(50, 250) * 0.10
        + np.log1p(base["rvol"].fillna(0).clip(lower=0)) * 2
    )
    return base.sort_values(["score", "overlap", "rs3m"], ascending=False)


def ticker_persistence(rows, lookback_dates):
    sub = rows[rows["session_date"].isin(lookback_dates)].copy()
    bull = make_core_bullish(sub)
    if bull.empty:
        return pd.DataFrame()

    daily = bull.drop_duplicates(["session_date", "symbol"])
    p = (
        daily.groupby("symbol")
        .agg(
            days_present=("session_date", "nunique"),
            theme=("theme", "last"),
            rs1m=("rs1m", "median"),
            rs3m=("rs3m", "median"),
            rvol=("rvol", "median"),
        )
        .reset_index()
    )
    overlaps = bull.groupby("symbol")["source"].nunique().rename("scan_types").reset_index()
    p = p.merge(overlaps, on="symbol", how="left")

    tenema = (
        rows[
            (rows["session_date"].isin(lookback_dates))
            & (rows["scan_key"] == "trend10ema")
        ]
        .groupby("symbol")["session_date"]
        .nunique()
        .rename("10ema_days")
        .reset_index()
    )
    p = p.merge(tenema, on="symbol", how="left")
    p["10ema_days"] = p["10ema_days"].fillna(0).astype(int)
    return p.sort_values(["days_present", "rs3m", "10ema_days"], ascending=False)


def theme_status(current, history):
    if current.empty:
        return pd.DataFrame()

    hist_dates = sorted(history["session_date"].unique())
    prev_dates = hist_dates[-5:-1] if len(hist_dates) >= 2 else []
    recs = []

    for _, r in current.iterrows():
        theme = r["theme"]
        today = int(r["stocks"])
        vals = []
        for d in prev_dates:
            t = theme_participation_for_date(history, d)
            match = t[t["theme"] == theme]
            vals.append(int(match["stocks"].iloc[0]) if not match.empty else 0)

        avg = float(np.mean(vals)) if vals else 0
        present = sum(v > 0 for v in vals)

        if present >= 3 and today >= max(8, 0.75 * avg):
            status = "Persistent"
        elif today >= max(10, 1.5 * avg) and present <= 2:
            status = "Emerging"
        elif avg >= 10 and today < 0.5 * avg:
            status = "Fading"
        elif today >= 8:
            status = "Active / episodic"
        else:
            status = "Selective"

        recs.append({
            "theme": theme,
            "today": today,
            "prior_4d_avg": round(avg, 1),
            "status": status,
        })

    return pd.DataFrame(recs).sort_values("today", ascending=False)


def chart_template(fig, height=430):
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=55, b=20),
        legend_title_text="",
    )
    return fig


st.set_page_config(page_title=APP_TITLE, page_icon="📈", layout="wide")
init_db()

st.title("📈 Market Command Center")
st.caption(
    "Upload the five Thinkorswim scans, save the trading day, and track breadth, "
    "theme persistence, RS quality, distribution, and recurring leaders."
)

tabs = st.tabs(["Upload & Run", "Dashboard", "Persistence", "Stock Explorer", "History"])

with tabs[0]:
    st.subheader("Daily scan upload")
    c1, c2 = st.columns([1, 1])
    with c1:
        session_date = st.date_input("Trading date", value=date.today())
    with c2:
        notes = st.text_input("Optional note", placeholder="e.g. weak tape / CPI / earnings week")

    uploaded_files = st.file_uploader(
        "Drop all 5 scan CSVs here",
        type=["csv"],
        accept_multiple_files=True,
        help="The app reads the Thinkorswim watchlist name inside each CSV and automatically identifies the scan."
    )

    uploads, unknown_files, duplicate_files = auto_assign_uploads(uploaded_files)

    if uploaded_files:
        st.markdown("#### Auto-detected scans")
        detected_rows = []
        for key, label in SCAN_KEYS.items():
            f = uploads.get(key)
            detected_rows.append({
                "Scan": label,
                "Detected file": f.name if f is not None else "— missing —"
            })
        st.dataframe(pd.DataFrame(detected_rows), use_container_width=True, hide_index=True)

        if unknown_files:
            st.warning(
                "Could not identify: " + ", ".join(f.name for f in unknown_files) +
                ". The app identifies scans mainly from the Thinkorswim watchlist name stored inside each CSV."
            )

        if duplicate_files:
            st.warning(
                "Duplicate scan types detected: " +
                ", ".join(f"{SCAN_KEYS[k]} ({f.name})" for k, f in duplicate_files)
            )

    st.info(
        "The original four-scan bullish universe is preserved. "
        "10EMA Trend is tracked as a separate confirmation layer so historical comparisons remain apples-to-apples."
    )

    if st.button("Run EOD Routine", type="primary", use_container_width=True):
        try:
            missing_core = [k for k in ["rotation", "dcr", "ep", "trend47"] if k not in uploads]
            if missing_core:
                raise ValueError(
                    "Missing required scan(s): " +
                    ", ".join(SCAN_KEYS[k] for k in missing_core)
                )
            if "trend10ema" not in uploads:
                st.warning("10EMA Trend was not detected. The original four-scan routine can still run.")

            parsed = save_session(session_date, uploads, notes)
            st.success(f"Saved {session_date.isoformat()} with {len(parsed)} scan files.")
            st.rerun()
        except Exception as e:
            st.exception(e)

with tabs[1]:
    dates_df = get_dates()
    if dates_df.empty:
        st.warning("No saved sessions yet. Upload your scans first.")
    else:
        selected = st.selectbox("Session", dates_df["session_date"].tolist(), index=0)
        rows = load_rows()
        day = rows[rows["session_date"] == selected]
        counts = session_counts(rows)
        today_counts = counts[counts["date"] == selected].iloc[0]

        a, b, c, d, e, f = st.columns(6)
        a.metric("4% Rotation", int(today_counts["4% Rotation"]))
        b.metric("High DCR", int(today_counts["High DCR"]))
        c.metric("EP+", int(today_counts["Positive $9M EP"]))
        d.metric("EP−", int(today_counts["Negative $9M EP"]))
        e.metric("4/7 Trend", int(today_counts["4/7 Trend"]))
        f.metric("10EMA Trend", int(today_counts["10EMA Trend"]))

        st.markdown("### Market breadth")
        long_counts = counts.melt(
            id_vars="date",
            value_vars=["4% Rotation", "High DCR", "Positive $9M EP", "4/7 Trend", "10EMA Trend"],
            var_name="Scan",
            value_name="Count",
        )
        fig = px.line(long_counts, x="date", y="Count", color="Scan", markers=True)
        st.plotly_chart(chart_template(fig), use_container_width=True)

        left, right = st.columns(2)

        with left:
            st.markdown("### Today’s bullish theme participation")
            tp = theme_participation_for_date(rows, selected)
            if not tp.empty:
                fig = px.bar(
                    tp.sort_values("stocks", ascending=True),
                    x="stocks", y="theme", orientation="h",
                    labels={"stocks": "Unique stocks", "theme": ""}
                )
                st.plotly_chart(chart_template(fig, 480), use_container_width=True)

        with right:
            st.markdown("### Theme RS quality")
            rq = rs_quality(rows, selected)
            if not rq.empty:
                rq_plot = rq.sort_values("rs3m", ascending=True)
                rq_long = rq_plot.melt(
                    id_vars=["theme", "stocks"],
                    value_vars=["rs1m", "rs3m"],
                    var_name="Metric",
                    value_name="RS",
                )
                rq_long["Metric"] = rq_long["Metric"].map(
                    {"rs1m": "Median RS 1M", "rs3m": "Median RS 3M"}
                )
                fig = px.bar(
                    rq_long,
                    x="RS", y="theme", color="Metric",
                    barmode="group", orientation="h",
                    labels={"theme": "", "RS": "Median RS"}
                )
                st.plotly_chart(chart_template(fig, 480), use_container_width=True)

        st.markdown("### Theme persistence")
        default_themes = [
            "Healthcare / Biotech",
            "Consumer / Retail",
            "Software",
            "Semis / AI Hardware",
            "Energy",
        ]
        persist = theme_persistence(rows, default_themes)
        if not persist.empty:
            fig = px.line(
                persist, x="date", y="stocks", color="theme", markers=True,
                labels={"stocks": "Unique bullish participants", "date": "", "theme": "Theme"}
            )
            st.plotly_chart(chart_template(fig), use_container_width=True)

        st.markdown("### Positive $9M EP vs distribution")
        ep = day[day["scan_key"] == "ep"].copy()
        p1, p2 = st.columns(2)
        with p1:
            st.write("**Positive EP**")
            pos = ep[ep["pct_change"] > 0].sort_values("pct_change", ascending=False)
            st.dataframe(
                pos[["symbol", "theme", "pct_change", "rvol", "rs1m", "rs3m"]],
                use_container_width=True, hide_index=True
            )
        with p2:
            st.write("**Negative EP / distribution**")
            neg = ep[ep["pct_change"] < 0].sort_values("pct_change")
            st.dataframe(
                neg[["symbol", "theme", "pct_change", "rvol", "rs1m", "rs3m"]],
                use_container_width=True, hide_index=True
            )

        st.markdown("### Multi-scan leaders")
        ov = overlap_table(rows, selected)
        if not ov.empty:
            st.dataframe(
                ov[["symbol", "theme", "overlap", "10ema_confirm", "scans",
                    "pct_change", "rvol", "rs1m", "rs3m"]].head(30),
                use_container_width=True, hide_index=True
            )

        st.markdown("### Top 20 for chart review")
        top20 = ov.head(20)["symbol"].tolist() if not ov.empty else []
        if top20:
            st.code(",".join(top20), language=None)

        st.markdown("### Theme status")
        ts = theme_status(tp, rows[rows["session_date"] <= selected]) if not tp.empty else pd.DataFrame()
        if not ts.empty:
            st.dataframe(ts, use_container_width=True, hide_index=True)

with tabs[2]:
    rows = load_rows()
    dates = sorted(rows["session_date"].unique()) if not rows.empty else []
    if not dates:
        st.warning("No saved sessions yet.")
    else:
        st.subheader("Persistence")
        window = st.radio("Lookback", [5, 10, 20, 40], horizontal=True, index=1)
        lookback = dates[-window:]

        tickers = ticker_persistence(rows, lookback)
        if not tickers.empty:
            st.markdown(f"### Recurring leaders — last {len(lookback)} saved sessions")
            st.dataframe(
                tickers[["symbol", "theme", "days_present", "10ema_days",
                         "scan_types", "rs1m", "rs3m", "rvol"]].head(100),
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("### Theme persistence frequency")
        bull = make_core_bullish(rows[rows["session_date"].isin(lookback)])
        if not bull.empty:
            daily_theme = (
                bull.drop_duplicates(["session_date", "theme", "symbol"])
                .groupby(["session_date", "theme"])["symbol"].nunique()
                .rename("stocks")
                .reset_index()
            )
            summary = (
                daily_theme.groupby("theme")
                .agg(
                    days_present=("session_date", "nunique"),
                    avg_participation=("stocks", "mean"),
                    max_participation=("stocks", "max"),
                )
                .reset_index()
                .sort_values(["days_present", "avg_participation"], ascending=False)
            )
            st.dataframe(summary, use_container_width=True, hide_index=True)

with tabs[3]:
    rows = load_rows()
    if rows.empty:
        st.warning("No saved sessions yet.")
    else:
        symbols = sorted(rows["symbol"].unique())
        symbol = st.selectbox("Ticker", symbols)
        s = rows[rows["symbol"] == symbol].copy().sort_values(["session_date", "scan_key"])
        st.subheader(symbol)

        show = s[["session_date", "scan_key", "theme", "pct_change", "rvol", "rs1m", "rs3m"]].copy()
        show["scan"] = show["scan_key"].map(SCAN_KEYS)
        st.dataframe(
            show[["session_date", "scan", "theme", "pct_change", "rvol", "rs1m", "rs3m"]],
            use_container_width=True,
            hide_index=True,
        )

        daily_presence = (
            s.assign(present=1)
            .pivot_table(index="session_date", columns="scan_key", values="present", aggfunc="max", fill_value=0)
            .reset_index()
        )
        st.markdown("### Scan presence over time")
        st.dataframe(daily_presence, use_container_width=True, hide_index=True)

with tabs[4]:
    rows = load_rows()
    dates_df = get_dates()
    if dates_df.empty:
        st.warning("No saved sessions yet.")
    else:
        st.subheader("Saved sessions")
        counts = session_counts(rows)
        hist = dates_df.merge(counts, left_on="session_date", right_on="date", how="left")
        st.dataframe(hist.drop(columns=["date"]), use_container_width=True, hide_index=True)

        st.markdown("### Delete / replace a session")
        st.caption("Uploading the same trading date again replaces that date’s scan rows.")
        delete_date = st.selectbox("Delete date", dates_df["session_date"].tolist(), key="delete_date")
        if st.button("Delete selected session"):
            conn = get_conn()
            conn.execute("DELETE FROM scan_rows WHERE session_date = ?", (delete_date,))
            conn.execute("DELETE FROM sessions WHERE session_date = ?", (delete_date,))
            conn.commit()
            conn.close()
            st.success(f"Deleted {delete_date}.")
            st.rerun()
