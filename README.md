
# Market Command Center

A private Streamlit web app for the daily EOD scan routine.

## What Version 1 does

- Uploads five Thinkorswim CSV scans:
  1. 4% Rotation
  2. High DCR
  3. $9M EP
  4. 4/7 Trend
  5. 10EMA Trend
- Saves each trading day in a local SQLite database.
- Keeps the original four-scan bullish-universe methodology intact.
- Uses the fifth 10EMA scan as a separate short-term confirmation layer.
- Tracks:
  - breadth history
  - positive vs negative $9M EP
  - theme participation
  - theme persistence
  - RS quality
  - multi-scan overlaps
  - ticker persistence
  - 10EMA confirmation
  - Top 20 chart-review candidates
- RS Quality chart is automatically sorted strongest-to-weakest by 3-month RS.

## Run locally

1. Install Python 3.10+.
2. Open a terminal in this folder.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Start the app:

```bash
streamlit run app.py
```

5. Your browser will open the app, usually at:

`http://localhost:8501`

The SQLite database (`market_command_center.db`) will be created in the same folder.

## Hosting later

This project can be deployed to Streamlit Community Cloud, Render, Railway, or another private host. For permanent cloud storage, the local SQLite database should later be replaced with a hosted database such as Postgres/Supabase.

## CSV expectations

The app expects Thinkorswim exports with a header beginning with:

`Symbol,Sub-Industry,...`

It automatically finds the header even when Thinkorswim adds metadata rows above it.

Expected fields when available:

- `Symbol`
- `Sub-Industry`
- `jeffRvol.c`
- `jeffRS1month.c`
- `jeffRS3month.c`
- `ADR`
- `jeff$VolBillion`
- `jy%change`

## Important methodology choice

The original bullish-universe count remains:

- 4% Rotation
- High DCR
- 4/7 Trend
- Positive $9M EP only

The new 10EMA Trend scan is displayed and used as a confirmation/ranking layer, but it does not silently change historical theme-participation definitions.
