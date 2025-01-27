import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import davies_bouldin_score
import matplotlib.pyplot as plt

# Load datasets
customers = pd.read_csv("Customers.csv")  
transactions = pd.read_csv("Transactions.csv")  
products = pd.read_csv("Products.csv")  

# Merge datasets
merged_data = transactions.merge(customers, on="CustomerID").merge(products, on="ProductID")

# Calculate Price if missing
if 'Price' not in merged_data.columns:
    merged_data['Price'] = merged_data['TotalValue'] / merged_data['Quantity']

# Preprocess for clustering
cluster_data = merged_data.groupby("CustomerID").agg({
    "TotalValue": "sum",
    "Quantity": "sum",
    "Price": "mean"
}).reset_index()

# Scale the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(cluster_data[["TotalValue", "Quantity", "Price"]])

# Apply KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=42)
cluster_data["Cluster"] = kmeans.fit_predict(scaled_data)

# Evaluate clustering
db_index = davies_bouldin_score(scaled_data, kmeans.labels_)
print(f"Davies-Bouldin Index: {db_index}")

# Visualize clusters using PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

plt.figure(figsize=(10, 6))
plt.scatter(pca_data[:, 0], pca_data[:, 1], c=cluster_data["Cluster"], cmap="viridis", s=50)
plt.title("Customer Clusters")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label="Cluster")
plt.show()
