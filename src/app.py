import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go
import os

# ── Page Config ────────────────────────────────────────
st.set_page_config(
    page_title="Predict Home",
    page_icon="🏠",
    layout="wide"
)

# ── Load Model & Data ──────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load('model/predict_home_model.pkl')

@st.cache_data
def load_data():
    return pd.read_csv('data/clean_data.csv')

@st.cache_data
def load_encoders():
    return joblib.load('model/encoders.pkl')

@st.cache_data
def load_area_stats():
    with open('model/area_stats.json') as f:
        return json.load(f)

model    = load_model()
df       = load_data()
encoders = load_encoders()
area_stats = load_area_stats()

# ── Header ─────────────────────────────────────────────
st.title("🏠 Predict Home")
st.markdown("*Delhi NCR Real Estate Price Intelligence*")
st.divider()

# ── Tabs ───────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🔮 Price Predictor",
    "📊 Market Dashboard",
    "📍 Zone Insights"
])

# ══════════════════════════════════════════════════════
# TAB 1: PRICE PREDICTOR
# ══════════════════════════════════════════════════════
with tab1:
    st.subheader("Predict Property Price")
    st.markdown("Fill in the property details below:")

    col1, col2 = st.columns(2)

    with col1:
        area = st.selectbox(
            "📍 Zone / Area",
            ['Gurgaon', 'New Delhi', 'Noida',
             'Greater Noida', 'Ghaziabad', 'Faridabad']
        )
        bedroom = st.selectbox("🛏️ BHK", [1, 2, 3, 4, 5, 6])
        floor   = st.number_input("🏢 Floor Number", min_value=0, max_value=35, value=2)
        bathroom = st.selectbox("🚿 Bathrooms", [1, 2, 3, 4, 5])

    with col2:
        status = st.selectbox(
            "🏗️ Status",
            ['Ready to Move', 'Under Construction']
        )
        transaction = st.selectbox(
            "🔄 Transaction Type",
            ['Resale', 'New Property']
        )
        # Auto-fill rate per sqft based on zone
        default_rate = int(area_stats.get(area, 8000))
        use_custom_rate = st.checkbox(
            "📝 I know the exact rate per sqft"
        )
        if use_custom_rate:
            rate_per_sqft = st.number_input(
                "💰 Rate per sqft (₹)",
                min_value=1000,
                max_value=100000,
                value=default_rate,
                step=500
            )
        else:
            rate_per_sqft = default_rate
            st.info(
                f"💡 Using zone median: ₹{default_rate:,}/sqft "
                f"(based on {area} market data)"
            )
        

    if st.button("🔮 Predict Price", type="primary"):

        # Encode inputs
        status_enc      = encoders['status'].transform([status])[0]
        transaction_enc = encoders['transaction'].transform([transaction])[0]
        area_enc        = encoders['area'].transform([area])[0]

        # Area stats
        area_median = area_stats.get(area, 8000)
        ppsf_vs_area = rate_per_sqft / area_median

        # Build feature array
        features = pd.DataFrame([{
            'status':           status_enc,
            'floor':            floor,
            'transaction':      transaction_enc,
            'bathroom':         bathroom,
            'Rate_per_sqft':    rate_per_sqft,
            'bedroom':          bedroom,
            'area':             area_enc,
            'area_median_ppsf': area_median,
            'ppsf_vs_area':     ppsf_vs_area
        }])

        prediction = model.predict(features)[0]
        low  = prediction * 0.85
        high = prediction * 1.15

        # Results
        st.success(f"### 🏠 Estimated Price: ₹{prediction:.1f} Lakhs")

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Low Estimate",  f"₹{low:.1f}L")
        col_b.metric("Best Estimate", f"₹{prediction:.1f}L", delta="Predicted")
        col_c.metric("High Estimate", f"₹{high:.1f}L")

        st.info(
            f"💡 Median Rate/sqft in **{area}**: ₹{area_median:,.0f} | "
            f"Your property is **{ppsf_vs_area:.2f}x** the zone median"
        )

        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode  = "gauge+number",
            value = prediction,
            title = {'text': "Predicted Price (₹ Lakhs)"},
            gauge = {
                'axis': {'range': [0, 500]},
                'bar':  {'color': "#1f77b4"},
                'steps': [
                    {'range': [0,   100], 'color': "#90EE90"},
                    {'range': [100, 250], 'color': "#FFD700"},
                    {'range': [250, 500], 'color': "#FF6347"}
                ]
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════
# TAB 2: MARKET DASHBOARD
# ══════════════════════════════════════════════════════
with tab2:
    st.subheader("Delhi NCR Market Overview")

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Listings", f"{len(df):,}")
    col2.metric("Avg Price",      f"₹{df['Price_Lakhs'].mean():.0f}L")
    col3.metric("Median Price",   f"₹{df['Price_Lakhs'].median():.0f}L")
    col4.metric("Zones Covered",  "6")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            df, x='Price_Lakhs',
            title='Price Distribution (Lakhs)',
            nbins=60,
            color_discrete_sequence=['#1f77b4']
        )
        fig.update_layout(
            xaxis_title="Price (Lakhs)",
            yaxis_title="Number of Properties"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        zone_median = df.groupby('area')['Price_Lakhs'].median().reset_index()
        zone_median.columns = ['Zone', 'Median Price (Lakhs)']
        fig = px.bar(
            zone_median,
            x='Zone', y='Median Price (Lakhs)',
            title='Median Price by Zone',
            color='Median Price (Lakhs)',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)

    # BHK distribution
    bhk_counts = df['bedroom'].value_counts().sort_index().reset_index()
    bhk_counts.columns = ['BHK', 'Count']
    fig = px.bar(
        bhk_counts, x='BHK', y='Count',
        title='Listings by BHK Type',
        color_discrete_sequence=['#2ca02c']
    )
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════
# TAB 3: ZONE INSIGHTS
# ══════════════════════════════════════════════════════
with tab3:
    st.subheader("Deep Dive into Any Zone")

    zones = ['Gurgaon', 'New Delhi', 'Noida',
             'Greater Noida', 'Ghaziabad', 'Faridabad']
    selected_zone = st.selectbox("Choose a Zone", zones)

    zone_df = df[df['area'] == selected_zone]

    if len(zone_df) > 0:
        col1, col2, col3 = st.columns(3)
        col1.metric("Listings",      len(zone_df))
        col2.metric("Median Price",  f"₹{zone_df['Price_Lakhs'].median():.0f}L")
        col3.metric("Median/sqft",   f"₹{zone_df['Rate_per_sqft'].median():.0f}")

        col1, col2 = st.columns(2)

        with col1:
            fig = px.box(
                zone_df, x='bedroom', y='Price_Lakhs',
                title=f'Price by BHK in {selected_zone}',
                labels={'bedroom': 'BHK', 'Price_Lakhs': 'Price (Lakhs)'}
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.scatter(
                zone_df,
                x='Rate_per_sqft', y='Price_Lakhs',
                color='bedroom',
                title=f'Rate/sqft vs Price in {selected_zone}',
                labels={
                    'Rate_per_sqft': 'Rate per sqft (₹)',
                    'Price_Lakhs': 'Price (Lakhs)'
                }
            )
            st.plotly_chart(fig, use_container_width=True)

# ── Footer ─────────────────────────────────────────────
st.divider()
st.markdown(
    "*Built by Saksham | Predict Home | "
    "Powered by Random Forest + Azure + Streamlit*"
)