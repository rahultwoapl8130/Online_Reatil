import streamlit as st
import pandas as pd
import joblib

kmeans_model = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")
sim_df = joblib.load("product_similarity.pkl")
# ===========================================
# Load saved files (run your notebook pipeline first
# so these files exist in the same folder as this app)
# ===========================================

# Confirmed mapping from your cluster profiling
cluster_map = {
    0: "High-Value",
    1: "Regular",
    2: "At-Risk"
}

# Same traffic-light theme used in your notebook charts
segment_colors = {
    "High-Value": "green",
    "Regular": "#cca300",   # darker yellow, more readable than pure yellow
    "At-Risk": "red"
}

# ===========================================
# Page setup
# ===========================================
st.set_page_config(page_title="Shopper Spectrum", layout="centered")
st.title("🛒 Shopper Spectrum")
st.caption("Customer Segmentation & Product Recommendations")

tab1, tab2 = st.tabs(["🔁 Product Recommendations", "👥 Customer Segmentation"])

# ===========================================
# TAB 1 — Product Recommendation
# ===========================================
with tab1:
    st.subheader("Find similar products")
    product = st.text_input("Enter a product name (exact match from dataset)")

    if st.button("Get Recommendations"):
        if product in sim_df.columns:
            top5 = sim_df[product].sort_values(ascending=False)[1:6]
            st.success(f"Top 5 products similar to **{product}**:")
            for name, score in top5.items():
                st.write(f"- **{name}**  (similarity: {score:.2f})")
        else:
            st.error("Product not found. Check the exact spelling from the dataset.")

# ===========================================
# TAB 2 — Customer Segmentation
# ===========================================
with tab2:
    st.subheader("Predict customer segment")

    recency = st.number_input("Recency (days since last purchase)", min_value=0, value=30)
    frequency = st.number_input("Frequency (number of orders)", min_value=0, value=3)
    monetary = st.number_input("Monetary (total amount spent)", min_value=0.0, value=500.0)

    if st.button("Predict Cluster"):
        X = scaler.transform([[recency, frequency, monetary]])
        cluster_id = kmeans_model.predict(X)[0]
        segment = cluster_map[cluster_id]
        color = segment_colors[segment]

        st.markdown(
            f"""
            <div style="padding:15px; border-radius:10px; background-color:{color}; color:white; text-align:center;">
                <h3>Predicted Segment: {segment}</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Short explanation per segment
        descriptions = {
            "High-Value": "Recent, frequent, big spender — prioritize loyalty & VIP offers.",
            "Regular": "Steady but average activity — good target for upsell campaigns.",
            "At-Risk": "Long time since last purchase, low activity — needs a win-back offer."
        }
        st.info(descriptions[segment])
