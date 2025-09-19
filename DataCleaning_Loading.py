import pandas as pd
import numpy as np

file_path = "retail_sales_dataset.csv"
df = pd.read_csv(file_path)

df.drop_duplicates(inplace=True)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date'])
df['Gender'] = df['Gender'].replace({0: "Male", 1: "Female"})
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Product Category'] = df['Product Category'].fillna("Unknown")
df['Quantity'] = df['Quantity'].fillna(0).astype(int)
df['Price per Unit'] = df['Price per Unit'].fillna(df['Price per Unit'].median())
df['Total Amount'] = df['Quantity'] * df['Price per Unit']
df['Age Group'] = pd.cut(df['Age'], bins=[0, 25, 40, 100], labels=["<25", "25-40", "40+"])

print("Shape:", df.shape)
print("Null Values:\n", df.isnull().sum())
print("Age Group Distribution:\n", df['Age Group'].value_counts())
print("Gender Distribution:\n", df['Gender'].value_counts())

df.to_csv("cleaned_transactions.csv", index=False)
print("Cleaned dataset saved as 'cleaned_transactions.csv'")
