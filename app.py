
import io
import sqlite3
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from supabase import create_client

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


def using_supabase():
    return (
        "supabase" in st.secrets
        and "url" in st.secrets["supabase"]
        and "service_role_key" in st.secrets["supabase"]
    )


def get_supabase():
    if not using_supabase():
        return None
    # Read current settings at call time, then cache by URL and key. This
    # prevents Streamlit from reusing a client created with an older secret.
    return _get_supabase_client(
        st.secrets["supabase"]["url"].strip(),
        st.secrets["supabase"]["service_role_key"].strip(),
    )


@st.cache_resource
def _get_supabase_client(url: str, key: str):
    return create_client(url, key)


def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db():
    # Supabase tables are created once with the SQL provided separately.
    if using_supabase():
        return
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
            avg_volume_30d REAL,
            pct_change REAL,
            theme TEXT,
            PRIMARY KEY (session_date, scan_key, symbol)
        );

        CREATE INDEX IF NOT EXISTS idx_scan_rows_date ON scan_rows(session_date);
        CREATE INDEX IF NOT EXISTS idx_scan_rows_symbol ON scan_rows(symbol);
        CREATE INDEX IF NOT EXISTS idx_scan_rows_theme ON scan_rows(theme);
        """
    )
    existing_columns = {
        row[1] for row in conn.execute("PRAGMA table_info(scan_rows)").fetchall()
    }
    if "avg_volume_30d" not in existing_columns:
        conn.execute("ALTER TABLE scan_rows ADD COLUMN avg_volume_30d REAL")
    conn.commit()
    conn.close()


def _nan_to_none(v):
    if pd.isna(v):
        return None
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v


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

    d = session_date.isoformat()

    if using_supabase():
        sb = get_supabase()

        sb.table("sessions").upsert(
            {"session_date": d, "notes": notes},
            on_conflict="session_date"
        ).execute()

        # Replace this trading day's rows atomically enough for this use case:
        sb.table("scan_rows").delete().eq("session_date", d).execute()

        records = []
        for scan_key, df in parsed.items():
            for _, r in df.iterrows():
                records.append({
                    "session_date": d,
                    "scan_key": scan_key,
                    "symbol": str(r["Symbol"]),
                    "sub_industry": str(r["Sub-Industry"]),
                    "rvol": _nan_to_none(r["rvol"]),
                    "rs1m": _nan_to_none(r["rs1m"]),
                    "rs3m": _nan_to_none(r["rs3m"]),
                    "adr": _nan_to_none(r["adr"]),
                    "dollar_vol_b": _nan_to_none(r["dollar_vol_b"]),
                    "avg_volume_30d": _nan_to_none(r["avg_volume_30d"]),
                    "pct_change": _nan_to_none(r["pct_change"]),
                    "theme": str(r["theme"]),
                })

        # Insert in chunks to keep requests manageable.
        for i in range(0, len(records), 500):
            sb.table("scan_rows").insert(records[i:i+500]).execute()

        return parsed

    conn = get_conn()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO sessions(session_date, notes) VALUES (?, ?)",
            (d, notes),
        )
        conn.execute("DELETE FROM scan_rows WHERE session_date = ?", (d,))

        rows = []
        for scan_key, df in parsed.items():
            for _, r in df.iterrows():
                rows.append((
                    d,
                    scan_key,
                    r["Symbol"],
                    r["Sub-Industry"],
                    r["rvol"],
                    r["rs1m"],
                    r["rs3m"],
                    r["adr"],
                    r["dollar_vol_b"],
                    r["avg_volume_30d"],
                    r["pct_change"],
                    r["theme"],
                ))
        conn.executemany(
            """
            INSERT OR REPLACE INTO scan_rows
            (session_date, scan_key, symbol, sub_industry, rvol, rs1m, rs3m, adr,
             dollar_vol_b, avg_volume_30d, pct_change, theme)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
    finally:
        conn.close()
    return parsed


def load_rows(start=None, end=None):
    if using_supabase():
        sb = get_supabase()
        q = sb.table("scan_rows").select("*")
        if start:
            q = q.gte("session_date", str(start))
        if end:
            q = q.lte("session_date", str(end))
        data = q.order("session_date").execute().data or []
        df = pd.DataFrame(data)
    else:
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

    if df.empty:
        return pd.DataFrame(columns=[
            "session_date", "scan_key", "symbol", "sub_industry",
            "rvol", "rs1m", "rs3m", "adr", "dollar_vol_b",
            "avg_volume_30d", "pct_change", "theme"
        ])

    for c in ["rvol", "rs1m", "rs3m", "adr", "dollar_vol_b", "avg_volume_30d", "pct_change"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def get_dates():
    if using_supabase():
        sb = get_supabase()
        data = (
            sb.table("sessions")
            .select("session_date,notes")
            .order("session_date", desc=True)
            .execute()
            .data or []
        )
        return pd.DataFrame(data)

    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT session_date, notes FROM sessions ORDER BY session_date DESC", conn
    )
    conn.close()
    return df

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

def parse_average_volume(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip().replace(",", "")
    if not text or text.lower() in {"nan", "none"}:
        return np.nan
    multiplier = 1
    suffix = text[-1].upper()
    if suffix in {"K", "M", "B"}:
        multiplier = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}[suffix]
        text = text[:-1].strip()
    try:
        return float(text) * multiplier
    except ValueError:
        return pd.to_numeric(text, errors="coerce")


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
        "avg_volume_30d": [
            "AvgVol30D", "Avg Vol 30D", "Avg Volume 30D", "Avg Volume (30D)",
            "30D Avg Volume", "30 Day Avg Volume", "30-Day Average Volume",
            "Average Volume 30D", "Average Volume (30 Days)", "30 Day Average Volume",
            "avg_volume_30d",
        ],
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

    if "avg_volume_30d" not in df.columns:
        df["avg_volume_30d"] = np.nan
    df["avg_volume_30d"] = df["avg_volume_30d"].map(parse_average_volume)

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


def theme_participation_for_period(rows, session_dates):
    period_rows = rows[rows["session_date"].isin(session_dates)]
    bull = make_core_bullish(period_rows)
    if bull.empty:
        return pd.DataFrame(
            columns=["theme", "avg_stocks_per_session", "days_active", "ticker_days"]
        )

    daily = (
        bull.drop_duplicates(["session_date", "theme", "symbol"])
        .groupby(["session_date", "theme"])["symbol"]
        .nunique()
        .rename("stocks")
        .reset_index()
    )
    summary = (
        daily.groupby("theme")
        .agg(
            ticker_days=("stocks", "sum"),
            days_active=("session_date", "nunique"),
        )
        .reset_index()
    )
    session_count = max(len(session_dates), 1)
    summary["avg_stocks_per_session"] = summary["ticker_days"] / session_count
    return summary.sort_values(
        ["avg_stocks_per_session", "days_active", "ticker_days"],
        ascending=False,
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


def multiscan_persistence(rows, lookback_dates):
    daily_frames = []
    for session_date in lookback_dates:
        leaders = overlap_table(rows, session_date)
        if leaders.empty:
            continue
        leaders = leaders[leaders["overlap"] >= 2].copy()
        if leaders.empty:
            continue
        leaders["session_date"] = session_date
        leaders["10ema_hit"] = leaders["10ema_confirm"].astype(int)
        daily_frames.append(leaders)

    if not daily_frames:
        return pd.DataFrame()

    daily = pd.concat(daily_frames, ignore_index=True)
    summary = (
        daily.groupby("symbol")
        .agg(
            multiscan_days=("session_date", "nunique"),
            total_scan_appearances=("overlap", "sum"),
            max_scans=("overlap", "max"),
            theme=("theme", "last"),
            rs1m=("rs1m", "median"),
            rs3m=("rs3m", "median"),
            rvol=("rvol", "median"),
            tenema_days=("10ema_hit", "sum"),
        )
        .reset_index()
    )
    return summary.sort_values(
        ["multiscan_days", "total_scan_appearances", "rs3m"],
        ascending=False,
        na_position="last",
    )


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


def selected_plotly_point(event):
    if isinstance(event, dict):
        selection = event.get("selection", {})
        points = selection.get("points", []) if isinstance(selection, dict) else []
    else:
        selection = getattr(event, "selection", None)
        points = getattr(selection, "points", []) if selection is not None else []
    return points[-1] if points else None


def theme_members(day, theme, fallback_metric="rs3m", limit=20):
    bullish = make_core_bullish(day)
    matched = bullish[bullish["theme"] == theme].copy()
    if matched.empty:
        return pd.DataFrame(), False
    if "avg_volume_30d" not in matched.columns:
        matched["avg_volume_30d"] = np.nan

    members = (
        matched.groupby("symbol", as_index=False)
        .agg(
            theme=("theme", "first"),
            rs1m=("rs1m", "median"),
            rs3m=("rs3m", "median"),
            rvol=("rvol", "median"),
        )
    )
    latest_volume = (
        matched[matched["avg_volume_30d"].notna()]
        .sort_values("session_date")
        .drop_duplicates("symbol", keep="last")[["symbol", "avg_volume_30d"]]
    )
    members = members.merge(latest_volume, on="symbol", how="left")
    days_present = (
        matched.drop_duplicates(["session_date", "symbol"])
        .groupby("symbol")["session_date"]
        .nunique()
        .rename("days_present")
        .reset_index()
    )
    members = members.merge(days_present, on="symbol", how="left")
    volume_available = members["avg_volume_30d"].notna().any()
    if volume_available:
        members = members[members["avg_volume_30d"].notna()]
        sort_column = "avg_volume_30d"
    else:
        sort_column = fallback_metric
    members = members.sort_values(
        sort_column, ascending=False, na_position="last"
    ).head(limit)
    return members, volume_available


def render_theme_members(day, theme, context, fallback_metric="rs3m"):
    members, volume_available = theme_members(day, theme, fallback_metric=fallback_metric)
    st.markdown(f"#### {theme} — {context}")
    if members.empty:
        st.info("No bullish scan members were saved for this theme on this session.")
        return

    if volume_available:
        st.caption("Top 20 ranked by 30-day average volume.")
    else:
        st.caption(
            "No 30-day average volume is saved for this period yet. "
            "Showing the top 20 by relative strength until your CSVs include "
            "an AvgVol30D or Avg Volume 30D column."
        )

    display = members[["symbol", "days_present", "avg_volume_30d", "rs1m", "rs3m", "rvol"]].rename(
        columns={
            "symbol": "Ticker",
            "days_present": "Sessions Present",
            "avg_volume_30d": "30D Avg Volume",
            "rs1m": "RS 1M",
            "rs3m": "RS 3M",
            "rvol": "RVOL",
        }
    )
    st.dataframe(display, use_container_width=True, hide_index=True)
    st.caption("Use the copy icon to copy these comma-separated tickers.")
    st.code(",".join(members["symbol"].tolist()), language=None)


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

if using_supabase():
    st.success("Cloud history: connected")
else:
    st.warning("Cloud history not connected yet. Using temporary local storage.")

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
            st.markdown("### Bullish theme participation")
            participation_window = st.radio(
                "Period",
                ["Today", "1 Week", "1 Month", "3 Months"],
                horizontal=True,
                key="dashboard_theme_participation_window",
            )
            if participation_window == "Today":
                participation_dates = [selected]
                participation_rows = day
                tp = theme_participation_for_date(rows, selected)
                tp_plot_base = tp.assign(strength=tp["stocks"]) if not tp.empty else tp
                strength_label = "Bullish participants"
                period_caption = "Today’s unique bullish scan participants."
            else:
                session_lengths = {"1 Week": 5, "1 Month": 20, "3 Months": 60}
                available_dates = sorted(
                    rows.loc[rows["session_date"] <= selected, "session_date"].unique()
                )
                participation_dates = available_dates[-session_lengths[participation_window]:]
                participation_rows = rows[rows["session_date"].isin(participation_dates)]
                tp = theme_participation_for_period(rows, participation_dates)
                tp_plot_base = tp.rename(
                    columns={"avg_stocks_per_session": "strength"}
                ) if not tp.empty else tp
                strength_label = "Avg participants / session"
                period_caption = (
                    f"Average unique bullish participants per saved session across "
                    f"the last {len(participation_dates)} sessions through {selected}."
                )

            st.caption(
                "Warmer bars indicate stronger breadth. Click a bar to see the theme’s "
                "ticker list for the selected period. " + period_caption
            )
            if not tp_plot_base.empty:
                tp_plot = tp_plot_base.sort_values("strength", ascending=True)
                fig = px.bar(
                    tp_plot,
                    x="strength", y="theme", orientation="h", color="strength",
                    color_continuous_scale="YlOrRd",
                    range_color=(0, max(1, float(tp_plot["strength"].max()))),
                    custom_data=["theme"],
                    hover_data={
                        "days_active": True,
                        "ticker_days": True,
                        "strength": ":.1f",
                        "theme": False,
                    } if participation_window != "Today" else None,
                    labels={"strength": strength_label, "theme": "", "color": strength_label}
                )
                participation_event = st.plotly_chart(
                    chart_template(fig, 480),
                    use_container_width=True,
                    key=f"dashboard_theme_participation_{participation_window}",
                    on_select="rerun",
                    selection_mode=["points"],
                )
                participation_point = selected_plotly_point(participation_event)
                if participation_point:
                    custom = participation_point.get("customdata") or []
                    participation_theme = participation_point.get("y")
                    if not participation_theme and custom:
                        participation_theme = custom[0]
                    if participation_theme:
                        context = f"{participation_window} bullish participation"
                        render_theme_members(
                            participation_rows,
                            participation_theme,
                            context,
                            fallback_metric="rs3m",
                        )

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
                    custom_data=["theme", "Metric"],
                    labels={"theme": "", "RS": "Median RS"}
                )
                rs_event = st.plotly_chart(
                    chart_template(fig, 480),
                    use_container_width=True,
                    key="dashboard_theme_rs_quality",
                    on_select="rerun",
                    selection_mode=["points"],
                )
                rs_point = selected_plotly_point(rs_event)
                if rs_point:
                    custom = rs_point.get("customdata") or []
                    rs_theme = rs_point.get("y")
                    if not rs_theme and custom:
                        rs_theme = custom[0]
                    rs_metric = custom[1] if len(custom) > 1 else None
                    if rs_theme:
                        metric_sort = "rs1m" if rs_metric == "Median RS 1M" else "rs3m"
                        metric_label = rs_metric or "Median RS 3M"
                        render_theme_members(
                            day, rs_theme, metric_label, fallback_metric=metric_sort,
                        )

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
        multiscan_window = st.radio(
            "Lookback saved sessions",
            [5, 10, 15, 30],
            horizontal=True,
            key="dashboard_multiscan_lookback",
        )
        dashboard_history = rows[rows["session_date"] <= selected]
        history_dates = sorted(dashboard_history["session_date"].unique())[-multiscan_window:]
        multiscan = multiscan_persistence(rows, history_dates)
        st.caption(
            f"Looking across the last {len(history_dates)} saved sessions through {selected}. "
            "A multi-scan day means a ticker appeared in at least two bullish scans."
        )
        if not multiscan.empty:
            st.dataframe(
                multiscan[["symbol", "theme", "multiscan_days", "total_scan_appearances",
                           "max_scans", "tenema_days", "rs1m", "rs3m", "rvol"]],
                use_container_width=True,
                hide_index=True,
            )
            st.caption("Copy the ranked ticker list:")
            st.code(",".join(multiscan["symbol"].tolist()), language=None)
        else:
            st.info("No multi-scan leaders found in this lookback.")

        st.markdown("### Top 20 for chart review")
        daily_ov = overlap_table(rows, selected)
        top20 = daily_ov.head(20)["symbol"].tolist() if not daily_ov.empty else []
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
            chart_rows = tickers.head(20).copy()
            chart_rows["persistence_rate"] = chart_rows["days_present"] / len(lookback)
            chart_rows["session_label"] = chart_rows["days_present"].astype(str) + f"/{len(lookback)} sessions"
            chart_rows = chart_rows.sort_values("days_present", ascending=True)
            fig = px.bar(
                chart_rows,
                x="days_present",
                y="symbol",
                orientation="h",
                color="persistence_rate",
                color_continuous_scale="YlOrRd",
                range_color=(0, 1),
                text="session_label",
                custom_data=["theme", "persistence_rate"],
                labels={
                    "days_present": "Saved sessions present",
                    "symbol": "",
                    "persistence_rate": "Share of sessions",
                },
                hover_data=["theme", "10ema_days", "scan_types", "rs1m", "rs3m"],
            )
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_layout(
                height=max(360, 28 * len(chart_rows) + 100),
                margin=dict(l=20, r=100, t=55, b=20),
                xaxis_range=[0, max(int(chart_rows["days_present"].max()) + 2, 3)],
            )
            st.caption("Warmer color means the ticker appeared in a larger share of the selected sessions.")
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                tickers[["symbol", "theme", "days_present", "10ema_days",
                         "scan_types", "rs1m", "rs3m", "rvol"]].head(100),
                use_container_width=True,
                hide_index=True,
            )

            st.caption("Copy the symbols shown above for the selected lookback window.")
            persistence_symbols = tickers.head(100)["symbol"].tolist()
            st.code(",".join(persistence_symbols), language=None)

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
            if using_supabase():
                sb = get_supabase()
                sb.table("scan_rows").delete().eq("session_date", delete_date).execute()
                sb.table("sessions").delete().eq("session_date", delete_date).execute()
            else:
                conn = get_conn()
                conn.execute("DELETE FROM scan_rows WHERE session_date = ?", (delete_date,))
                conn.execute("DELETE FROM sessions WHERE session_date = ?", (delete_date,))
                conn.commit()
                conn.close()
            st.success(f"Deleted {delete_date}.")
            st.rerun()
