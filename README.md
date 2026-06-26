# 🛒 Shopper Spectrum: Customer Segmentation & Product Recommendations

A complete end-to-end machine learning project that analyzes e-commerce transaction data to segment customers using RFM analysis and KMeans clustering, and recommends products using item-based collaborative filtering — deployed as a live web app on Google Cloud Run.

**🔗 Live App:** [
https://online-reatil-582440781606.europe-west1.run.app]

---

## 📌 Overview

The global e-commerce industry generates massive volumes of transaction data daily. This project turns that raw data into two actionable business tools:

1. **Customer Segmentation** — classifies customers into behavioral segments (High-Value, Regular, At-Risk) based on Recency, Frequency, and Monetary (RFM) analysis
2. **Product Recommendation Engine** — suggests similar products using collaborative filtering, enabling cross-sell and personalization

## 🧠 Problem Type

- Unsupervised Machine Learning – Clustering
- Collaborative Filtering – Recommendation System

## 🗂 Dataset

Online Retail transaction dataset — 541,909 rows, 8 columns (InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country).

## 🔍 Key Findings

- **Three natural customer segments emerged** from RFM clustering (validated via Elbow Method + Silhouette Score, k=3, silhouette score = 0.453):
  - **High-Value** (~820 customers): recent purchases, high order frequency, highest spend — priority for loyalty programs
  - **Regular** (~1,880 customers): the largest segment, moderate activity across all RFM dimensions
  - **At-Risk** (~900 customers): long gap since last purchase, low frequency and spend — needs retention campaigns

- **Frequency and Monetary are strongly correlated (0.72)** — customers who order more often also tend to spend more in total, confirming RFM's three dimensions capture complementary (not redundant) behavior signals

- **KMeans outperformed Agglomerative Clustering** (silhouette: 0.453 vs 0.422) and was selected as the production model — it also supports real-time prediction on new customer inputs, unlike Agglomerative

- **Product similarity via cosine similarity** successfully surfaced meaningful cross-sell relationships (e.g., party/craft-supply products clustering together), validated through a similarity heatmap

## 🛠 Tech Stack

| Category | Tools |
|---|---|
| Data Processing | Python, Pandas, NumPy |
| Machine Learning | Scikit-learn (KMeans, Agglomerative Clustering, StandardScaler) |
| Visualization | Matplotlib, Seaborn |
| Recommendation Engine | Cosine Similarity (Collaborative Filtering) |
| Web App | Streamlit |
| Deployment | Docker, Google Cloud Run |
| Version Control | Git, GitHub |

## 📊 Methodology

1. **Data Cleaning** — removed missing CustomerID, cancelled invoices, and invalid quantity/price records
2. **EDA** — analyzed transaction volume by country, top-selling products, and purchase trends over time
3. **Feature Engineering** — built customer-level RFM table; removed outliers using the IQR method
4. **Clustering** — standardized RFM values, compared KMeans vs. Agglomerative Clustering using Elbow Method and Silhouette Score, selected optimal k=3
5. **Segmentation** — profiled and labeled clusters based on RFM averages (High-Value, Regular, At-Risk)
6. **Recommendation System** — built a customer-product matrix and computed cosine similarity between products
7. **Deployment** — containerized with Docker and deployed to Google Cloud Run with continuous deployment from GitHub

## 🚀 Running Locally

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Repository Structure

```
.
├── app.py                      # Streamlit web application
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container configuration for deployment
├── kmeans_model.pkl            # Trained KMeans clustering model
├── scaler.pkl                  # StandardScaler for RFM feature scaling
├── product_similarity.pkl      # Precomputed product similarity matrix
└── online_Reatail.ipynb        # Full analysis notebook (EDA, clustering, evaluation)
```

## 📈 App Features

**Customer Segmentation Tab**
Enter Recency, Frequency, and Monetary values → instantly predicts the customer's segment with a color-coded result (🟢 High-Value, 🟡 Regular, 🔴 At-Risk) and a tailored business recommendation.

**Product Recommendation Tab**
Enter any product name → returns the top 5 most similar products based on purchase pattern similarity, with similarity scores.

## 🎯 Business Applications

- Targeted marketing campaigns by customer segment
- Personalized product recommendations to boost cross-sell
- Early identification of at-risk customers for retention efforts
- Data-driven inventory and demand planning

---

*Built as part of a customer analytics project applying unsupervised learning and collaborative filtering to real-world e-commerce data.*
