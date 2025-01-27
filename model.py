from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import pandas as pd
import csv

# Load datasets
customers = pd.read_csv("Customers.csv")  
transactions = pd.read_csv("Transactions.csv")  
products = pd.read_csv("Products.csv")  

# Preprocessing
customer_features = customers.merge(transactions, on="CustomerID").merge(products, on="ProductID")
customer_features = customer_features.groupby("CustomerID").agg({
    "Region": "first",
    "Category": lambda x: " ".join(x),
    "TotalValue": "sum"
}).reset_index()

# Encode the Region column
label_encoder = LabelEncoder()
customer_features["Region"] = label_encoder.fit_transform(customer_features["Region"])

# Encode the Category column
customer_features["Category"] = customer_features["Category"].astype("category").cat.codes

# Create feature vectors
features = customer_features[["Region", "Category", "TotalValue"]]

# Scale the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Compute cosine similarity
similarity_matrix = cosine_similarity(scaled_features)

# Generate Lookalikes
lookalike_results = {}
for i in range(20):  # For the first 20 customers
    similarities = list(enumerate(similarity_matrix[i]))
    sorted_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)[1:4]
    lookalike_results[customer_features.iloc[i]["CustomerID"]] = [
        (customer_features.iloc[j[0]]["CustomerID"], j[1]) for j in sorted_similarities
    ]

# Save results to CSV
with open("FirstName_LastName_Lookalike.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["cust_id", "lookalikes"])
    for key, value in lookalike_results.items():
        writer.writerow([key, value])

print("Lookalike results saved to FirstName_LastName_Lookalike.csv")
