import os

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st


st.set_page_config(
    page_title="UIDAI Biometric Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
)


UT_NAMES = {
    "Andaman And Nicobar Islands",
    "Chandigarh",
    "Dadra And Nagar Haveli And Daman And Diu",
    "Delhi",
    "Jammu & Kashmir",
    "Ladakh",
    "Lakshadweep",
    "Puducherry",
}

STATE_LIST = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka",
    "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram",
    "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana",
    "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal", "Andaman And Nicobar Islands",
    "Chandigarh", "Dadra And Nagar Haveli And Daman And Diu", "Delhi", "Jammu & Kashmir",
    "Ladakh", "Lakshadweep", "Puducherry",
]


st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=DM+Serif+Display:ital@0;1&display=swap');

:root {
  --bg-1: #f4efe6;
  --bg-2: #e7f0ea;
  --ink: #172026;
  --muted: #5d6972;
  --card: rgba(255, 255, 255, 0.62);
  --stroke: rgba(23, 32, 38, 0.12);
  --accent: #0f766e;
  --accent-2: #b45309;
}

html, body, [class*="css"] {
  font-family: 'Space Grotesk', sans-serif;
  color: var(--ink);
}

.stApp {
  background:
    radial-gradient(1200px 500px at 5% 0%, rgba(180, 83, 9, 0.13), transparent 62%),
    radial-gradient(900px 420px at 95% 10%, rgba(15, 118, 110, 0.18), transparent 62%),
    linear-gradient(130deg, var(--bg-1), var(--bg-2));
}

.hero {
  border: 1px solid var(--stroke);
  background: linear-gradient(120deg, rgba(255,255,255,0.82), rgba(255,255,255,0.52));
  backdrop-filter: blur(10px);
  border-radius: 20px;
  padding: 1.2rem 1.2rem;
  margin-bottom: 1rem;
  box-shadow: 0 8px 30px rgba(20, 35, 30, 0.08);
}

.hero h1 {
  font-family: 'DM Serif Display', serif;
  line-height: 1.06;
  letter-spacing: 0.2px;
  margin: 0;
  font-size: clamp(1.8rem, 4vw, 3rem);
}

.hero p {
  margin: 0.6rem 0 0 0;
  color: var(--muted);
}

.metric-card {
  border: 1px solid var(--stroke);
  background: var(--card);
  border-radius: 14px;
  padding: 0.9rem;
  box-shadow: 0 6px 16px rgba(0, 0, 0, 0.04);
}

.metric-label {
  font-size: 0.82rem;
  color: var(--muted);
}

.metric-value {
  font-size: 1.4rem;
  font-weight: 700;
}

@media (max-width: 768px) {
  .hero {
    padding: 1rem;
  }
}
</style>
""",
    unsafe_allow_html=True,
)


def _read_secret(key_name: str, default: str = "") -> str:
    if key_name in st.secrets:
        return str(st.secrets[key_name])
    return os.getenv(key_name, default)


AI_ENDPOINT = _read_secret("AISTAL_API_ENDPOINT", "https://api.aistal.com/v1/chat/completions")
AI_MODEL = _read_secret("AISTAL_MODEL", "gpt-4o-mini")
AI_KEY = _read_secret("AISTAL_API_KEY", "")


@st.cache_data(show_spinner=False)
def make_demo_data(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-03-01", "2025-12-29", freq="D")
    rows = []
    for state in STATE_LIST:
        district_count = 6 if state in UT_NAMES else 18
        for d_idx in range(district_count):
            district_name = f"District-{d_idx + 1}"
            base = rng.integers(20, 600)
            for date in dates:
                season_factor = 1.0 + 0.28 * np.sin((date.month - 1) / 12 * 2 * np.pi)
                weekday_factor = 0.85 if date.weekday() in (5, 6) else 1.0
                child = max(0, int(rng.normal(base * season_factor * weekday_factor * 0.47, 12)))
                adult = max(0, int(rng.normal(base * season_factor * weekday_factor * 0.53, 16)))
                rows.append(
                    {
                        "date": date,
                        "state": state,
                        "district": district_name,
                        "pincode": str(100000 + rng.integers(1, 899999)),
                        "bio_age_5_17": child,
                        "bio_age_17_": adult,
                    }
                )
    df = pd.DataFrame(rows)
    return df


def normalize_and_prepare(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    col_map = {c: c.strip().lower().replace(" ", "_") for c in df.columns}
    df = df.rename(columns=col_map)

    alias_map = {
        "bio_age_5_17": ["bio_age_5_17", "child_updates", "child", "bio_age_5_to_17"],
        "bio_age_17_": ["bio_age_17_", "adult_updates", "adult", "bio_age_18_plus", "bio_age_18"],
        "date": ["date", "update_date"],
        "state": ["state", "state_ut", "state_or_ut"],
        "district": ["district", "district_name"],
        "pincode": ["pincode", "pin", "postal_code"],
    }

    for target, aliases in alias_map.items():
        if target in df.columns:
            continue
        found = next((a for a in aliases if a in df.columns), None)
        if found:
            df[target] = df[found]

    required = ["date", "state", "district", "pincode", "bio_age_5_17", "bio_age_17_"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["date"]).copy()

    for num_col in ["bio_age_5_17", "bio_age_17_"]:
        num_series = pd.to_numeric(df[num_col], errors="coerce")
        num_series = num_series.fillna(0).clip(lower=0)
        df[num_col] = num_series

    df["state"] = df["state"].astype(str).str.strip().str.title()
    df["district"] = df["district"].astype(str).str.strip().str.title()
    df["pincode"] = df["pincode"].astype(str).str.strip()

    df = (
        df.groupby(["date", "state", "district", "pincode"], as_index=False)[["bio_age_5_17", "bio_age_17_"]]
        .sum()
    )

    df["total_bio_updates"] = df["bio_age_5_17"] + df["bio_age_17_"]
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["month_label"] = df["date"].dt.to_period("M").astype(str)
    df["weekday"] = df["date"].dt.day_name()
    df["region_type"] = np.where(df["state"].isin(UT_NAMES), "Union Territory", "State")
    return df


def generate_ai_insight(
    model: str,
    endpoint: str,
    api_key: str,
    filtered_df: pd.DataFrame,
) -> str:
    national = filtered_df.groupby("date", as_index=False)["total_bio_updates"].sum()
    state_totals = (
        filtered_df.groupby("state")["total_bio_updates"]
        .sum()
        .reset_index()
        .sort_values(by="total_bio_updates", ascending=False)
    )
    top_states = state_totals.head(8).to_dict(orient="records")

    summary_block = {
        "rows": int(filtered_df.shape[0]),
        "start_date": str(filtered_df["date"].min().date()),
        "end_date": str(filtered_df["date"].max().date()),
        "total_updates": int(filtered_df["total_bio_updates"].sum()),
        "child_share": round(float(filtered_df["bio_age_5_17"].sum() / max(filtered_df["total_bio_updates"].sum(), 1)), 4),
        "adult_share": round(float(filtered_df["bio_age_17_"].sum() / max(filtered_df["total_bio_updates"].sum(), 1)), 4),
        "daily_average": round(float(national["total_bio_updates"].mean()), 2),
        "daily_std": round(float(national["total_bio_updates"].std(ddof=0)), 2),
        "top_states": top_states,
    }

    prompt = (
        "You are an India public-sector data analyst. "
        "Give compact real-time insights with 5 bullets only. "
        "Each bullet must include one metric value and one policy action. "
        "Tone: executive, no fluff. Data summary: "
        f"{summary_block}"
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return concise analytics only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    resp = requests.post(endpoint, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


st.markdown(
    """
<div class="hero">
  <h1>UIDAI Biometric Intelligence Command Center</h1>
  <p>Premium analytical dashboard built from your notebook logic: state inequality, district hotspots, seasonality, anomaly signals, and priority-focus states.</p>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Control Deck")
    source_mode = st.radio("Data source", ["Demo data", "Upload CSV"], horizontal=False)
    uploaded_file = st.file_uploader("Upload cleaned UIDAI CSV", type=["csv"]) if source_mode == "Upload CSV" else None

    st.subheader("Filters")

if source_mode == "Upload CSV" and uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
else:
    raw_df = make_demo_data()

df = pd.DataFrame()
try:
    df = normalize_and_prepare(raw_df)
except Exception as exc:
    st.error(f"Data processing error: {exc}")
    st.stop()

with st.sidebar:
    min_date = df["date"].min().date()
    max_date = df["date"].max().date()
    date_window = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    selected_states = st.multiselect("State / UT", sorted(df["state"].unique()), default=[])
    near_zero_threshold = st.slider("Near-zero threshold (daily updates)", min_value=0, max_value=50, value=5)

if isinstance(date_window, tuple) and len(date_window) == 2:
    start_date, end_date = date_window
else:
    start_date = min_date
    end_date = max_date
f_df = df[(df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)].copy()
if selected_states:
    f_df = f_df[f_df["state"].isin(selected_states)]

if f_df.empty:
    st.warning("No rows available after filters. Adjust date/state filters.")
    st.stop()

k1, k2, k3, k4 = st.columns(4)
metric_data = [
    ("Total Updates", f"{int(f_df['total_bio_updates'].sum()):,}"),
    ("States / UTs", f"{f_df['state'].nunique()}"),
    ("Districts", f"{f_df['district'].nunique()}"),
    ("Date Span", f"{f_df['date'].min().date()} to {f_df['date'].max().date()}"),
]
for col, (label, value) in zip([k1, k2, k3, k4], metric_data):
    col.markdown(
        f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div></div>",
        unsafe_allow_html=True,
    )

state_age = (
    f_df.groupby("state", as_index=False)[["bio_age_5_17", "bio_age_17_", "total_bio_updates"]]
    .sum()
    .sort_values(by="total_bio_updates", ascending=False)
)

monthly = f_df.groupby("month_label", as_index=False)[["bio_age_5_17", "bio_age_17_", "total_bio_updates"]].sum()
weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
weekday = f_df.groupby("weekday", as_index=False)["total_bio_updates"].sum()
weekday["weekday"] = pd.Categorical(weekday["weekday"], categories=weekday_order, ordered=True)
weekday = weekday.sort_values(by="weekday")

district_totals = (
    f_df.groupby(["state", "district"], as_index=False)["total_bio_updates"]
    .sum()
    .sort_values(by="total_bio_updates", ascending=False)
)
district_totals["district_state"] = district_totals["district"] + " (" + district_totals["state"] + ")"

near_zero_state = (
    f_df[f_df["total_bio_updates"] <= near_zero_threshold]
    .groupby("state", as_index=False)
    .size()
    .rename(columns={"size": "near_zero_days"})
    .sort_values(by="near_zero_days", ascending=False)
)

national_daily = f_df.groupby("date", as_index=False)["total_bio_updates"].sum().sort_values(by="date")
national_daily["rolling_mean"] = national_daily["total_bio_updates"].rolling(window=3, min_periods=2).mean()
national_daily["rolling_std"] = national_daily["total_bio_updates"].rolling(window=3, min_periods=2).std().fillna(0)
national_daily["upper"] = national_daily["rolling_mean"] + 2 * national_daily["rolling_std"]
national_daily["lower"] = (national_daily["rolling_mean"] - 2 * national_daily["rolling_std"]).clip(lower=0)
national_daily["is_anomaly"] = (national_daily["total_bio_updates"] > national_daily["upper"]) | (
    national_daily["total_bio_updates"] < national_daily["lower"]
)

state_stats = f_df.groupby("state", as_index=False).agg(
    total_updates=("total_bio_updates", "sum"),
    district_count=("district", "nunique"),
)
state_near = f_df[f_df["total_bio_updates"] <= near_zero_threshold].groupby("state", as_index=False).size()
state_near = state_near.rename(columns={"size": "near_zero_days"})
state_stats = state_stats.merge(state_near, on="state", how="left").fillna({"near_zero_days": 0})
state_stats["low_volume_score"] = 1 - (state_stats["total_updates"] / state_stats["total_updates"].max())
state_stats["near_zero_score"] = state_stats["near_zero_days"] / max(state_stats["near_zero_days"].max(), 1)
state_stats["priority_score"] = 0.6 * state_stats["low_volume_score"] + 0.4 * state_stats["near_zero_score"]
priority_states = state_stats.sort_values("priority_score", ascending=False).head(10)

r1c1, r1c2 = st.columns(2)
with r1c1:
    fig = px.bar(
        state_age.head(10),
        x="state",
        y=["bio_age_5_17", "bio_age_17_"],
        barmode="group",
        title="Child vs Adult Aadhaar Biometric Updates (Top 10 States)",
        labels={"value": "Total Updates", "variable": "Age Group", "state": "State"},
        color_discrete_sequence=["#0f766e", "#b45309"],
    )
    fig.update_layout(height=460)
    st.plotly_chart(fig, width="stretch")

with r1c2:
    region_summary = f_df.groupby("region_type", as_index=False)["total_bio_updates"].sum()
    fig = px.bar(
        region_summary,
        x="region_type",
        y="total_bio_updates",
        title="Aadhaar Biometric Updates: States vs Union Territories",
        color="region_type",
        color_discrete_sequence=["#0f766e", "#b45309"],
    )
    fig.update_layout(showlegend=False, height=460)
    st.plotly_chart(fig, width="stretch")

r2c1, r2c2 = st.columns(2)
with r2c1:
    fig = px.bar(
        state_age.head(10),
        x="state",
        y="total_bio_updates",
        title="Top 10 States by Aadhaar Biometric Updates",
        color="total_bio_updates",
        color_continuous_scale="Tealgrn",
    )
    fig.update_layout(height=420, coloraxis_showscale=False)
    st.plotly_chart(fig, width="stretch")

with r2c2:
    fig = px.bar(
        state_age.tail(10),
        x="state",
        y="total_bio_updates",
        title="Bottom 10 States / UTs by Aadhaar Biometric Updates",
        color="total_bio_updates",
        color_continuous_scale="Oranges",
    )
    fig.update_layout(height=420, coloraxis_showscale=False)
    st.plotly_chart(fig, width="stretch")

r3c1, r3c2 = st.columns(2)
with r3c1:
    fig = px.bar(
        district_totals.head(10).sort_values("total_bio_updates"),
        x="total_bio_updates",
        y="district_state",
        orientation="h",
        title="Top 10 District Hotspots by Aadhaar Biometric Updates",
        color="total_bio_updates",
        color_continuous_scale="Tealgrn",
    )
    fig.update_layout(height=500, coloraxis_showscale=False)
    st.plotly_chart(fig, width="stretch")

with r3c2:
    fig = px.bar(
        district_totals.tail(10).sort_values("total_bio_updates"),
        x="total_bio_updates",
        y="district_state",
        orientation="h",
        title="Bottom 10 Districts by Aadhaar Updates",
        color="total_bio_updates",
        color_continuous_scale="Oranges",
    )
    fig.update_layout(height=500, coloraxis_showscale=False)
    st.plotly_chart(fig, width="stretch")

r4c1, r4c2 = st.columns(2)
with r4c1:
    fig = px.line(
        monthly,
        x="month_label",
        y=["bio_age_5_17", "bio_age_17_"],
        markers=True,
        title="Monthly Aadhaar Biometric Updates: Children vs Adults",
        labels={"value": "Total Updates", "variable": "Group", "month_label": "Month"},
        color_discrete_sequence=["#0f766e", "#b45309"],
    )
    fig.update_layout(height=430)
    st.plotly_chart(fig, width="stretch")

with r4c2:
    fig = px.line(
        monthly,
        x="month_label",
        y="total_bio_updates",
        markers=True,
        title="Total Monthly Aadhaar Biometric Updates (All Ages)",
        color_discrete_sequence=["#1f2937"],
    )
    fig.update_layout(height=430)
    st.plotly_chart(fig, width="stretch")

r5c1, r5c2 = st.columns(2)
with r5c1:
    fig = px.bar(
        weekday,
        x="weekday",
        y="total_bio_updates",
        title="Weekday-wise Aadhaar Biometric Update Activity",
        color="total_bio_updates",
        color_continuous_scale="Tealgrn",
    )
    fig.update_layout(height=420, coloraxis_showscale=False)
    st.plotly_chart(fig, width="stretch")

with r5c2:
    if near_zero_state.empty:
        st.info("No near-zero days found for current threshold and filters.")
    else:
        fig = px.bar(
            near_zero_state.head(15).sort_values("near_zero_days"),
            x="near_zero_days",
            y="state",
            orientation="h",
            title="States with Highest Near-Zero Aadhaar Activity",
            color="near_zero_days",
            color_continuous_scale="Oranges",
        )
        fig.update_layout(height=420, coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")

r6c1, r6c2 = st.columns(2)
with r6c1:
    fig = px.line(
        national_daily,
        x="date",
        y=["total_bio_updates", "rolling_mean"],
        title="Anomaly Detection in Aadhaar Biometric Updates",
        labels={"value": "Updates", "variable": "Series"},
        color_discrete_sequence=["#1f2937", "#0f766e"],
    )
    anomalies = national_daily[national_daily["is_anomaly"]]
    if not anomalies.empty:
        fig.add_scatter(
            x=anomalies["date"],
            y=anomalies["total_bio_updates"],
            mode="markers",
            marker={"size": 8, "color": "#dc2626"},
            name="Anomaly",
        )
    fig.update_layout(height=430)
    st.plotly_chart(fig, width="stretch")

with r6c2:
    fig = px.bar(
        priority_states.sort_values("priority_score"),
        x="priority_score",
        y="state",
        orientation="h",
        title="Top 10 States Requiring Government Focus",
        color="priority_score",
        color_continuous_scale="OrRd",
    )
    fig.update_layout(height=430, coloraxis_showscale=False)
    st.plotly_chart(fig, width="stretch")

st.subheader("AI Real-Time Insight Engine")
if source_mode == "Upload CSV" and uploaded_file is not None:
    ai_endpoint = _read_secret("AISTAL_API_ENDPOINT", "https://api.aistal.com/v1/chat/completions")
    ai_model = _read_secret("AISTAL_MODEL", "gpt-4o-mini")
    ai_key = _read_secret("AISTAL_API_KEY", "")

    if not ai_key.strip():
        st.warning("AI insights disabled: `AISTAL_API_KEY` missing in `.streamlit/secrets.toml`.")
    else:
        signature = "|".join(
            [
                uploaded_file.name,
                str(start_date),
                str(end_date),
                ",".join(sorted(selected_states)) if selected_states else "ALL",
                str(int(f_df["total_bio_updates"].sum())),
                str(int(f_df.shape[0])),
            ]
        )

        if st.session_state.get("ai_signature") != signature:
            st.session_state["ai_error"] = ""
            try:
                with st.spinner("CSV upload detect ho gaya - AI insights generate ho rahe hain..."):
                    st.session_state["ai_text"] = generate_ai_insight(ai_model, ai_endpoint, ai_key, f_df)
                    st.session_state["ai_signature"] = signature
            except Exception as exc:
                st.session_state["ai_error"] = str(exc)

        if st.session_state.get("ai_error"):
            st.error(f"AI insight request failed: {st.session_state['ai_error']}")
        elif st.session_state.get("ai_text"):
            st.success("Uploaded CSV se AI insights ready.")
            st.markdown(st.session_state["ai_text"])

        if st.button("Refresh AI Insights"):
            st.session_state["ai_error"] = ""
            try:
                with st.spinner("Latest filters ke saath AI insights refresh ho rahe hain..."):
                    st.session_state["ai_text"] = generate_ai_insight(ai_model, ai_endpoint, ai_key, f_df)
                    st.session_state["ai_signature"] = signature
            except Exception as exc:
                st.session_state["ai_error"] = str(exc)
                st.error(f"AI insight request failed: {exc}")
else:
    st.info("Koi bhi CSV upload karte hi AI insights yahin dashboard me auto-generate honge.")

st.caption("Built from notebook analytics logic with interactive filters and premium visual layout.")
