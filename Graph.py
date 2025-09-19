import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_age_group_means(file="cleaned_transactions.csv"):
    df = pd.read_csv(file)
    means = df.groupby("Age Group")["Total Amount"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(6,4))
    sns.barplot(data=means, x="Age Group", y="Total Amount", palette="viridis", ax=ax)
    ax.set_title("Average Purchase Amount by Age Group")
    ax.set_ylabel("Mean Total Amount")
    ax.set_xlabel("Age Group")
    return fig  

def plot_gender_boxplot(file="cleaned_transactions.csv"):
    df = pd.read_csv(file)

    fig, ax = plt.subplots(figsize=(6,4))
    sns.boxplot(data=df, x="Gender", y="Total Amount", palette="Set2", ax=ax)
    ax.set_title("Purchase Amount Distribution by Gender")
    ax.set_ylabel("Total Amount")
    ax.set_xlabel("Gender")
    return fig  

