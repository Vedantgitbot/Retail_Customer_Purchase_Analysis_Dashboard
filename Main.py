import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats import weightstats as stests

df = pd.read_csv("cleaned_transactions.csv")

def plot_age_group_means(file="cleaned_transactions.csv"):
    df = pd.read_csv(file)
    means = df.groupby("Age Group")["Total Amount"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(6,4))
    sns.barplot(data=means, x="Age Group", y="Total Amount", palette="viridis", ax=ax)
    return fig  

def plot_gender_boxplot(file="cleaned_transactions.csv"):
    df = pd.read_csv(file)
    fig, ax = plt.subplots(figsize=(6,4))
    sns.boxplot(data=df, x="Gender", y="Total Amount", palette="Set2", ax=ax)
    return fig

groups = [df.loc[df['Age Group'] == g, 'Total Amount'] for g in df['Age Group'].dropna().unique()]
anova_result = stats.f_oneway(*groups)
tukey = sm.stats.multicomp.pairwise_tukeyhsd(df['Total Amount'], df['Age Group'])

t_stat, t_pval = stats.ttest_1samp(df['Total Amount'], 50)

male = df[df['Gender'] == "Male"]["Total Amount"]
female = df[df['Gender'] == "Female"]["Total Amount"]
t2_stat, t2_pval = stats.ttest_ind(male, female, equal_var=False)

ztest, zpval = stests.ztest(male, female)

st.set_page_config(page_title="Retail Purchase Analysis", layout="wide")

st.title("Retail Customer Purchase Analysis Dashboard")

st.header("Dataset Overview")
st.dataframe(df.head(20))
st.write("**Shape:**", df.shape)
st.write("**Null Values:**")
st.write(df.isnull().sum())

st.header("Statistical Tests")
col1, col2 = st.columns(2)
with col1:
    st.subheader("ANOVA")
    st.write("F-statistic:", round(anova_result.statistic, 4))
    st.write("p-value:", round(anova_result.pvalue, 4))
    st.subheader("One-sample T-test (H0: mean = 50)")
    st.write("Statistic:", round(t_stat, 4), "| p-value:", round(t_pval, 4))
with col2:
    st.subheader("Two-sample T-test (Male vs Female)")
    st.write("Statistic:", round(t2_stat, 4), "| p-value:", round(t2_pval, 4))
    st.subheader("Z-test (Male vs Female)")
    st.write("Statistic:", round(ztest, 4), "| p-value:", round(zpval, 4))

st.subheader("Tukey HSD Post-hoc Test")
st.text(tukey.summary())

st.header("Visualizations")
col1, col2 = st.columns(2)
with col1:
    st.markdown("#### Average Purchase by Age Group")
    fig1 = plot_age_group_means()
    st.pyplot(fig1)
with col2:
    st.markdown("#### Purchase Distribution by Gender")
    fig2 = plot_gender_boxplot()
    st.pyplot(fig2)
