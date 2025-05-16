# -*- coding: utf-8 -*-
"""

@author: Aayush Garg
"""
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd

# Load the uploaded CSV file
#file_path = "/mnt/data/4dbe5667-7b6b-41d7-82af-211562424d9a_a0d71eebebaa5ba00b5d1af1dd96a3dd.csv"

df = pd.read_csv("C:/Users/shiti/Downloads/4dbe5667-7b6b-41d7-82af-211562424d9a_a0d71eebebaa5ba00b5d1af1dd96a3dd.csv")

# Display basic information and first few rows of the dataset
df.info(), df.head()

# Step 1: Data Cleaning
df_cleaned = df.copy()

# Convert date column to datetime
df_cleaned['CompanyRegistrationdate_date'] = pd.to_datetime(df_cleaned['CompanyRegistrationdate_date'], errors='coerce')

# Drop rows with any null values for simplicity
df_cleaned.dropna(inplace=True)

# Strip whitespaces and standardize case for some categorical columns
df_cleaned['CompanyStatus'] = df_cleaned['CompanyStatus'].str.strip().str.title()
df_cleaned['CompanyClass'] = df_cleaned['CompanyClass'].str.strip().str.title()

# Step 2: Data Visualization

# 1. Bar plot of top 10 most common Company States
plt.figure(figsize=(10, 5))
df_cleaned['CompanyStateCode'].value_counts().head(10).plot(kind='bar', color='skyblue')
plt.title('Top 10 States with Most Companies')
plt.ylabel('Number of Companies')
plt.xlabel('State Code')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Histogram of Authorized Capital
plt.figure(figsize=(10, 5))
sns.histplot(df_cleaned['AuthorizedCapital'], bins=50, kde=True, color='green')
plt.title('Distribution of Authorized Capital')
plt.xlabel('Authorized Capital')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# 3. Pie chart of Company Class distribution
class_counts = df_cleaned['CompanyClass'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('Set2'))
plt.title('Company Class Distribution')
plt.axis('equal')
plt.show()

# 4. Box plot of Paid-up Capital by Company Status
plt.figure(figsize=(12, 6))
sns.boxplot(x='CompanyStatus', y='PaidupCapital', data=df_cleaned)
plt.title('Paid-up Capital by Company Status')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 5. Correlation heatmap for numeric features
plt.figure(figsize=(8, 6))
corr = df_cleaned[['AuthorizedCapital', 'PaidupCapital', 'nic_code']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

# Step 3: Linear Regression to predict PaidupCapital using AuthorizedCapital
X = df_cleaned[['AuthorizedCapital']]
y = df_cleaned['PaidupCapital']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Output evaluation results and a sample plot
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
plt.xlabel('Actual Paid-up Capital')
plt.ylabel('Predicted Paid-up Capital')
plt.title('Linear Regression: Actual vs Predicted')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.tight_layout()
plt.show()

mse, r2
