import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# PAGE CONFIG (ONLY ONCE)
st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")

plt.switch_backend("Agg")

# LOAD DATA
df = pd.read_csv("../data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
st.write("Data Loaded:", df.shape)

st.title("📊 Customer Churn Data Dashboard")

st.markdown("""
This interactive dashboard visualizes the **Telco Customer Churn Dataset**  
used for Machine Learning prediction.
""")

# SIDEBAR
st.sidebar.header("DATASET INFORMATION")
if st.sidebar.checkbox("Show Raw Dataset"):
    st.dataframe(df)

# ---------------- CHURN DISTRIBUTION ----------------
st.header("CHURN DISTRIBUTION")
fig1, ax1 = plt.subplots(figsize=(4, 4), dpi=80)
df["Churn"].value_counts().plot.pie(
    autopct="%1.1f%%",
    colors=["lightgreen", "salmon"],
    explode=[0, 0.1],
    ax=ax1
)
ax1.set_ylabel("")
fig1.tight_layout()
st.pyplot(fig1, use_container_width=False)

# ---------------- GENDER VS CHURN ----------------
st.header("GENDER VS CHURN")
fig2, ax2 = plt.subplots(figsize=(4, 4), dpi=80)
sns.countplot(data=df, x="gender", hue="Churn", ax=ax2)
fig2.tight_layout()
st.pyplot(fig2, use_container_width=False)

# ---------------- INTERNET SERVICE ----------------
st.header("INTERNET SERVICE TYPE VS CHURN")
fig3, ax3 = plt.subplots(figsize=(4, 4), dpi=80)
sns.countplot(data=df, x="InternetService", hue="Churn", ax=ax3)
ax3.tick_params(axis="x", rotation=45)
fig3.tight_layout()
st.pyplot(fig3, use_container_width=False)

# ---------------- CONTRACT TYPE ----------------
st.header("CONTRACT TYPE VS CHURN")
fig4, ax4 = plt.subplots(figsize=(4, 4), dpi=80)
sns.countplot(data=df, x="Contract", hue="Churn", ax=ax4)
ax4.tick_params(axis="x", rotation=45)
fig4.tight_layout()
st.pyplot(fig4, use_container_width=False)

# ---------------- MONTHLY CHARGES ----------------
st.header("MONTHLY CHARGES DISTRIBUTION")
fig5, ax5 = plt.subplots(figsize=(4, 4), dpi=80)
sns.boxplot(data=df, x="Churn", y="MonthlyCharges", ax=ax5)
fig5.tight_layout()
st.pyplot(fig5, use_container_width=False)

# ---------------- TENURE ----------------
st.header("TENURE HISTOGRAM")
fig6, ax6 = plt.subplots(figsize=(4, 4), dpi=80)
sns.histplot(df["tenure"], kde=True, bins=30, ax=ax6)
fig6.tight_layout()
st.pyplot(fig6, use_container_width=False)

# ---------------- HEATMAP ----------------
st.header("Correlation Heatmap (Numerical Features)")
numeric_df = df.select_dtypes(include=["int64", "float64"])
fig7, ax7 = plt.subplots(figsize=(5, 4), dpi=80)
sns.heatmap(numeric_df.corr(), cmap="coolwarm", ax=ax7)
fig7.tight_layout()
st.pyplot(fig7, use_container_width=False)
