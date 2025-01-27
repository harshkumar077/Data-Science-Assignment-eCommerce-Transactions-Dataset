import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
customers = pd.read_csv("Customers.csv")
products = pd.read_csv("Products.csv")
transactions = pd.read_csv("Transactions.csv")

# EDA
# Check for missing values
print(customers.isnull().sum())
print(products.isnull().sum())
print(transactions.isnull().sum())

# Summary statistics
print(customers.describe())
print(products.describe())
print(transactions.describe())

# Merge datasets for analysis
merged_data = transactions.merge(customers, on="CustomerID").merge(products, on="ProductID")

# Top regions by revenue
region_revenue = merged_data.groupby("Region")["TotalValue"].sum().sort_values(ascending=False)
print(region_revenue)

# Visualize regional revenue
plt.figure(figsize=(10, 6))
region_revenue.plot(kind="bar", color="skyblue")
plt.title("Revenue by Region")
plt.ylabel("Total Revenue (USD)")
plt.xlabel("Region")
plt.show()

# Seasonal transaction trends
merged_data["TransactionDate"] = pd.to_datetime(merged_data["TransactionDate"])
merged_data["Month"] = merged_data["TransactionDate"].dt.month

monthly_sales = merged_data.groupby("Month")["TotalValue"].sum()
plt.figure(figsize=(10, 6))
monthly_sales.plot(kind="line", marker="o")
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Revenue (USD)")
plt.show()

# Popular product categories
category_sales = merged_data.groupby("Category")["TotalValue"].sum().sort_values(ascending=False)
print(category_sales)

# Visualize product category sales
plt.figure(figsize=(10, 6))
category_sales.plot(kind="bar", color="lightgreen")
plt.title("Sales by Product Category")
plt.xlabel("Category")
plt.ylabel("Total Revenue (USD)")
plt.show()
