"""
Andhra Pradesh House Price Predictor (Streamlit web app)
Models: Linear Regression vs Random Forest

Install:  pip install streamlit scikit-learn pandas numpy matplotlib
Run:      streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RANDOM_STATE = 42
CITIES = ["Visakhapatnam", "Vijayawada", "Amaravati", "Tirupati", "Guntur",
          "Kakinada", "Rajahmundry", "Nellore", "Kurnool"]


# ------------------------------------------------------------------
# DATA (simulated Andhra Pradesh housing data)
# City price levels are illustrative assumptions, NOT official figures.
# ------------------------------------------------------------------
def generate_ap_housing_data(n=3000, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    city_ppsf = {
        "Visakhapatnam": 5800, "Vijayawada": 5200, "Amaravati": 4600,
        "Tirupati": 4400, "Guntur": 4000, "Kakinada": 3700,
        "Rajahmundry": 3600, "Nellore": 3500, "Kurnool": 3100,
    }
    city_weights = [0.20, 0.18, 0.08, 0.10, 0.12, 0.07, 0.08, 0.10, 0.07]
    cities = rng.choice(list(city_ppsf), size=n, p=city_weights)
    property_type = rng.choice(["Apartment", "Independent House", "Villa"],
                               size=n, p=[0.62, 0.30, 0.08])
    bedrooms = rng.choice([1, 2, 3, 4, 5], size=n, p=[0.07, 0.38, 0.38, 0.14, 0.03])
    base_area = {1: 600, 2: 1000, 3: 1450, 4: 2000, 5: 2800}
    area = np.array([base_area[b] for b in bedrooms], dtype=float)
    area *= np.where(property_type == "Villa", 1.5,
                     np.where(property_type == "Independent House", 1.2, 1.0))
    area *= rng.normal(1.0, 0.15, n)
    area = np.clip(area, 350, 6000).round()
    bathrooms = np.clip(bedrooms + rng.integers(-1, 2, n), 1, 6)
    age_years = np.clip(rng.gamma(2.0, 4.0, n), 0, 40).round()
    floor = np.where(property_type == "Apartment", rng.integers(0, 15, n), 0)
    dist = np.clip(rng.gamma(2.5, 3.0, n), 0.5, 35).round(1)
    tier = rng.choice(["Premium", "Mid-range", "Developing"], size=n, p=[0.2, 0.5, 0.3])
    parking = rng.choice([0, 1, 2], size=n, p=[0.2, 0.55, 0.25])
    furnishing = rng.choice(["Unfurnished", "Semi-furnished", "Furnished"],
                            size=n, p=[0.5, 0.35, 0.15])
    near_hw = rng.choice([0, 1], size=n, p=[0.7, 0.3])
    rera = rng.choice([0, 1], size=n, p=[0.35, 0.65])

    ppsf = np.array([city_ppsf[c] for c in cities], dtype=float)
    ppsf *= pd.Series(tier).map({"Premium": 1.35, "Mid-range": 1.0, "Developing": 0.78}).to_numpy()
    ppsf *= np.exp(-0.035 * dist)
    ppsf *= np.exp(-0.012 * age_years)
    ppsf *= np.where(property_type == "Villa", 1.25,
                     np.where(property_type == "Independent House", 1.08, 1.0))
    ppsf *= 1 + 0.004 * np.minimum(floor, 10)
    ppsf *= 1 + 0.05 * parking
    ppsf *= 1 + 0.06 * near_hw
    ppsf *= 1 + 0.04 * rera
    ppsf *= pd.Series(furnishing).map(
        {"Unfurnished": 1.0, "Semi-furnished": 1.04, "Furnished": 1.09}).to_numpy()
    ppsf *= (area / 1200) ** -0.05
    ppsf *= rng.lognormal(0, 0.10, n)
    price = (ppsf * area / 1e5).round(2)

    df = pd.DataFrame({
        "city": cities, "locality_tier": tier, "property_type": property_type,
        "area_sqft": area, "bedrooms": bedrooms, "bathrooms": bathrooms,
        "age_years": age_years, "floor": floor, "distance_to_center_km": dist,
        "parking": parking, "furnishing": furnishing,
        "near_highway_or_metro": near_hw, "rera_approved": rera,
        "price_lakhs": price,
    })
    for col in ["age_years", "distance_to_center_km", "furnishing"]:
        df.loc[rng.random(n) < 0.03, col] = np.nan
    out_idx = rng.choice(n, size=15, replace=False)
    df.loc[out_idx, "price_lakhs"] *= rng.uniform(1.8, 2.5, 15)
    return df


# ------------------------------------------------------------------
# TRAINING (cached so it only runs once)
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="Training the two models (first run only)...")
def train_models():
    df = generate_ap_housing_data().drop_duplicates()
    df = df[(df["price_lakhs"] > 0) & (df["area_sqft"] > 0)]
    ppsf = df["price_lakhs"] * 1e5 / df["area_sqft"]
    q1, q3 = ppsf.quantile([0.25, 0.75])
    iqr = q3 - q1
    df = df[(ppsf >= q1 - 3 * iqr) & (ppsf <= q3 + 3 * iqr)].reset_index(drop=True)

    X = df.drop(columns="price_lakhs")
    y = df["price_lakhs"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE)

    cat_cols = ["city", "locality_tier", "property_type", "furnishing"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    def prep(scale):
        num_steps = [("impute", SimpleImputer(strategy="median"))]
        if scale:
            num_steps.append(("scale", StandardScaler()))
        return ColumnTransformer([
            ("num", Pipeline(num_steps), num_cols),
            ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                              ("onehot", OneHotEncoder(handle_unknown="ignore"))]),
             cat_cols),
        ])

    def wrap(model):
        return TransformedTargetRegressor(regressor=model, func=np.log1p,
                                          inverse_func=np.expm1)

    lin = wrap(Pipeline([("prep", prep(True)), ("model", LinearRegression())]))
    rf = wrap(Pipeline([("prep", prep(False)),
                        ("model", RandomForestRegressor(
                            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1))]))
    lin.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    rows = {}
    for name, m in [("Linear Regression", lin), ("Random Forest", rf)]:
        p = m.predict(X_test)
        rows[name] = {
            "MAE (lakhs)": mean_absolute_error(y_test, p),
            "RMSE (lakhs)": np.sqrt(mean_squared_error(y_test, p)),
            "R2": r2_score(y_test, p),
            "MAPE (%)": np.mean(np.abs((y_test - p) / y_test)) * 100,
        }
    return lin, rf, pd.DataFrame(rows).T.round(3), df, X.columns.tolist()


lin_model, rf_model, results, data, feature_order = train_models()

# ------------------------------------------------------------------
# PAGE
# ------------------------------------------------------------------
st.set_page_config(page_title="AP House Price Predictor", page_icon="🏠", layout="wide")
st.title("🏠 Andhra Pradesh House Price Predictor")
st.caption("Compares two machine learning models: Linear Regression and Random Forest")
st.warning(
    "Demo project: the models are trained on **simulated** housing data for "
    "Andhra Pradesh, so the prices shown are estimates for learning purposes, "
    "not real market valuations."
)

tab_predict, tab_compare, tab_data = st.tabs(
    ["🔮 Predict price", "📊 Model comparison", "🗂️ Explore data"])

# ---------------- Tab 1: prediction ----------------
with tab_predict:
    st.subheader("Enter house details")
    c1, c2, c3 = st.columns(3)
    with c1:
        city = st.selectbox("City", CITIES)
        locality = st.selectbox("Locality type", ["Premium", "Mid-range", "Developing"], index=1)
        ptype = st.selectbox("Property type", ["Apartment", "Independent House", "Villa"])
        area = st.slider("Area (sq ft)", 350, 6000, 1250, step=50)
    with c2:
        bedrooms = st.selectbox("Bedrooms (BHK)", [1, 2, 3, 4, 5], index=1)
        bathrooms = st.selectbox("Bathrooms", [1, 2, 3, 4, 5, 6], index=1)
        age = st.slider("Age of property (years)", 0, 40, 4)
        distance = st.slider("Distance from city center (km)", 0.5, 35.0, 6.0, step=0.5)
    with c3:
        floor = st.slider("Floor (apartments only)", 0, 14, 3,
                          disabled=(ptype != "Apartment"))
        parking = st.selectbox("Parking spots", [0, 1, 2], index=1)
        furnishing = st.selectbox("Furnishing", ["Unfurnished", "Semi-furnished", "Furnished"])
        near_hw = st.checkbox("Near highway / metro / major road", value=True)
        rera = st.checkbox("RERA approved", value=True)

    if st.button("Predict price", type="primary"):
        house = pd.DataFrame([{
            "city": city, "locality_tier": locality, "property_type": ptype,
            "area_sqft": area, "bedrooms": bedrooms, "bathrooms": bathrooms,
            "age_years": age, "floor": floor if ptype == "Apartment" else 0,
            "distance_to_center_km": distance, "parking": parking,
            "furnishing": furnishing, "near_highway_or_metro": int(near_hw),
            "rera_approved": int(rera),
        }])[feature_order]

        p_lin = float(lin_model.predict(house)[0])
        p_rf = float(rf_model.predict(house)[0])

        def fmt(lakhs):
            return f"₹{lakhs:,.1f} lakhs" if lakhs < 100 else f"₹{lakhs / 100:,.2f} crore"

        m1, m2, m3 = st.columns(3)
        m1.metric("Linear Regression", fmt(p_lin))
        m2.metric("Random Forest", fmt(p_rf))
        m3.metric("Average of both", fmt((p_lin + p_rf) / 2))
        st.caption(
            f"The two models differ by {abs(p_lin - p_rf):,.1f} lakhs "
            f"({abs(p_lin - p_rf) / ((p_lin + p_rf) / 2) * 100:.1f}%). "
            f"Typical error on unseen houses is about "
            f"{results.loc['Random Forest', 'MAE (lakhs)']:.1f} lakhs (Random Forest)."
        )

# ---------------- Tab 2: comparison ----------------
with tab_compare:
    st.subheader("How well do the models perform on unseen houses?")
    st.write("Scores on the 20% of houses the models never saw during training.")
    st.dataframe(results, use_container_width=True)
    st.markdown(
        "- **MAE**: average error in lakhs (lower is better)\n"
        "- **RMSE**: like MAE but punishes big mistakes more (lower is better)\n"
        "- **R²**: share of price variation explained (closer to 1 is better)\n"
        "- **MAPE**: average error as a percentage of the price (lower is better)"
    )
    st.bar_chart(results[["MAE (lakhs)", "RMSE (lakhs)"]])
    st.info(
        "**Linear Regression** adds up the effect of each feature (simple and "
        "easy to explain). **Random Forest** averages many decision trees and "
        "can capture more complex patterns (more flexible, harder to explain)."
    )

# ---------------- Tab 3: data ----------------
with tab_data:
    st.subheader("Explore the training data")
    st.write(f"{len(data):,} simulated listings across {data['city'].nunique()} cities.")
    st.write("**Median price by city (lakhs)**")
    st.bar_chart(data.groupby("city")["price_lakhs"].median().sort_values())
    st.write("**Sample of the data**")
    st.dataframe(data.sample(20, random_state=1), use_container_width=True)
