import streamlit as st
st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.switch_backend("Agg")

df = pd.read_csv("../data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
st.write("Data Loaded:", df.shape)



st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")

st.title("📊 Customer Churn Data Dashboard")

st.markdown("""
This interactive dashboard visualizes the **Telco Customer Churn Dataset**  
used for Machine Learning prediction.
""")


st.sidebar.header(" DARASET INFORMATION")
if st.sidebar.checkbox("Show Raw Dataset"):
    st.write(df)

st.header("CHURN DISTRIBUTION")
fig1, ax1 = plt.subplots()
df["Churn"].value_counts().plot.pie(
    autopct="%1.1f%%",
    colors=["lightgreen", "salmon"],
    explode=[0, 0.1],
    ax=ax1
)
ax1.set_ylabel("")
st.pyplot(fig1)


st.header(" GENDER VS CHURN")
fig2, ax2 = plt.subplots()
sns.countplot(data=df, x="gender", hue="Churn", ax=ax2)
st.pyplot(fig2)

st.header(" iNTERNET SERVICE TYPE VS CHURN")
fig3, ax3 = plt.subplots()
sns.countplot(data=df, x="InternetService", hue="Churn", ax=ax3)
plt.xticks(rotation=45)
st.pyplot(fig3)


st.header(" CONTRACT TYPE VS CHURN")
fig4, ax4 = plt.subplots()
sns.countplot(data=df, x="Contract", hue="Churn", ax=ax4)
plt.xticks(rotation=45)
st.pyplot(fig4)



st.header("MONTHLY CHARGES DISTRIBUTION")
fig5, ax5 = plt.subplots()
sns.boxplot(data=df, x="Churn", y="MonthlyCharges", ax=ax5)
st.pyplot(fig5)


st.header("TENURE HISTOGRAM")
fig6, ax6 = plt.subplots()
sns.histplot(df["tenure"], kde=True, bins=30, ax=ax6)
st.pyplot(fig6)

st.header("Correlation Heatmap (Numerical Features)")
numeric_df = df.select_dtypes(include=["int64","float64"])
fig7, ax7 = plt.subplots(figsize=(12,5))
sns.heatmap(numeric_df.corr(), annot=False, cmap="coolwarm", ax=ax7)
st.pyplot(fig7)
