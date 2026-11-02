"""Customer Segmentation with K-Means - Streamlit app."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

DEFAULT_DATA = Path(__file__).parent / "data" / "Test.csv"
REQUIRED_COLUMNS = [
    "Gender", "Ever_Married", "Age", "Graduated",
    "Work_Experience", "Spending_Score", "Family_Size",
]
SPENDING_MAP = {"Low": 0, "Average": 1, "High": 2}


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Fill numeric gaps with the median and categorical gaps with the mode."""
    clean = df.dropna(subset=["Gender", "Age", "Spending_Score"]).copy()
    clean = clean.reset_index(drop=True)
    for col in ["Work_Experience", "Family_Size"]:
        clean[col] = clean[col].fillna(clean[col].median())
    for col in ["Ever_Married", "Graduated"]:
        clean[col] = clean[col].fillna(clean[col].mode()[0])
    return clean


def build_features(clean: pd.DataFrame) -> pd.DataFrame:
    """Turn text columns into numbers so K-Means can measure distances."""
    return pd.DataFrame({
        "Gender": (clean["Gender"] == "Male").astype(int),
        "Ever_Married": (clean["Ever_Married"] == "Yes").astype(int),
        "Age": clean["Age"],
        "Graduated": (clean["Graduated"] == "Yes").astype(int),
        "Work_Experience": clean["Work_Experience"],
        "Spending_Score": clean["Spending_Score"].map(SPENDING_MAP),
        "Family_Size": clean["Family_Size"],
    })


@st.cache_data
def segment(df: pd.DataFrame, n_clusters: int):
    clean = clean_data(df)
    features = build_features(clean)
    scaled = StandardScaler().fit_transform(features)
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(scaled)
    coords = PCA(n_components=2, random_state=42).fit_transform(scaled)
    return clean, features, labels, coords


def main():
    st.set_page_config(page_title="Customer Segmentation", page_icon="📊", layout="wide")
    st.title("Customer Segmentation with K-Means")
    st.caption("Group customers into segments by age, family, work experience and spending behaviour.")

    with st.sidebar:
        st.header("Controls")
        uploaded = st.file_uploader("Upload your own CSV", type="csv")
        n_clusters = st.slider("Number of segments (K)", 2, 8, 4)
        st.markdown("**Expected columns**")
        st.code("\n".join(REQUIRED_COLUMNS))

    if uploaded is not None:
        df = pd.read_csv(uploaded)
    elif DEFAULT_DATA.exists():
        df = pd.read_csv(DEFAULT_DATA)
    else:
        st.info("Upload a CSV in the sidebar to get started.")
        st.stop()

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        st.error(f"Your file is missing these columns: {', '.join(missing)}")
        st.stop()

    clean, features, labels, coords = segment(df, n_clusters)
    profile_df = features.assign(Cluster=labels)

    c1, c2, c3 = st.columns(3)
    c1.metric("Customers", f"{len(clean):,}")
    c2.metric("Segments", n_clusters)
    c3.metric("Features used", features.shape[1])

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Segment sizes", "Segment profiles", "Cluster map", "Data & download"]
    )

    with tab1:
        st.bar_chart(pd.Series(labels).value_counts().sort_index())

    with tab2:
        st.subheader("Average feature values per segment")
        st.dataframe(profile_df.groupby("Cluster").mean().round(2), width="stretch")
        st.subheader("Spending score mix per segment")
        st.dataframe(
            pd.crosstab(labels, clean["Spending_Score"], normalize="index").round(2),
            width="stretch",
        )
        if "Profession" in clean.columns:
            st.subheader("Profession mix per segment")
            st.dataframe(
                pd.crosstab(labels, clean["Profession"], normalize="index").round(2),
                width="stretch",
            )

    with tab3:
        st.caption("Seven features squeezed into two axes with PCA so we can plot them.")
        fig, ax = plt.subplots(figsize=(8, 6))
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="viridis", s=20, alpha=0.7)
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.set_title("Customer segmentation using K-Means")
        fig.colorbar(sc, label="Cluster")
        st.pyplot(fig)

    with tab4:
        result = clean.assign(Cluster=labels)
        st.dataframe(result.head(50), width="stretch")
        st.download_button(
            "Download clustered data (CSV)",
            result.to_csv(index=False).encode("utf-8"),
            file_name="Test_clustered.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
