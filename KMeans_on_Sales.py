# ==============================
# K-Means & Hierarchical Clustering on Sales Data
# ==============================

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.metrics import silhouette_score

# -----------------------------
# Step 1: Load the dataset
# -----------------------------
df = pd.read_csv("sales_data_sample.csv", encoding='latin1')
print(" Dataset loaded successfully!")
print("Shape:", df.shape)
print(df.head())

# -----------------------------
# Step 2: Select numeric columns for clustering
# -----------------------------
num_df = df.select_dtypes(include=['number']).dropna()
print("\nSelected numeric columns for clustering:", num_df.columns.tolist())

# Standardize the numeric features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(num_df)

# -----------------------------
# Step 3: Elbow Method to find optimal K
# -----------------------------
inertia = []
K = range(1, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(K, inertia, 'bo-', linewidth=2, markersize=8)
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia (Within-Cluster Sum of Squares)")
plt.grid(True)
plt.show()

# -----------------------------
# Step 4: Apply K-Means clustering
# -----------------------------
optimal_k = 3  # based on elbow plot
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster_KMeans'] = kmeans.fit_predict(scaled_data)

print("\nK-Means Clustering Completed!")
print(df['Cluster_KMeans'].value_counts())

# Compute silhouette score for quality check
sil_score = silhouette_score(scaled_data, df['Cluster_KMeans'])
print("Silhouette Score (K-Means):", round(sil_score, 4))

# -----------------------------
# Step 5: Visualize K-Means clusters (using first 2 features)
# -----------------------------
plt.figure(figsize=(8,5))
plt.scatter(scaled_data[:, 0], scaled_data[:, 1],
            c=df['Cluster_KMeans'], cmap='viridis', s=40)
plt.title("K-Means Cluster Visualization (First 2 Features)")
plt.xlabel(num_df.columns[0])
plt.ylabel(num_df.columns[1])
plt.show()

# -----------------------------
# Step 6: Hierarchical Clustering
# -----------------------------
linked = linkage(scaled_data, method='ward')

plt.figure(figsize=(10,5))
dendrogram(linked, truncate_mode='level', p=5)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()

# Assign hierarchical clusters
df['Cluster_HC'] = fcluster(linked, t=optimal_k, criterion='maxclust')

print("\nHierarchical Clustering Completed!")
print(df['Cluster_HC'].value_counts())

