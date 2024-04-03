import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("forestfires.csv")

# Select relevant features for clustering
X = df[['X','Y','FFMC','DMC','DC','temp','RH','wind','rain','area']]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform K-means clustering
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_scaled)
clusters = kmeans.predict(X_scaled)

df['cluster'] = clusters

plt.figure(figsize=(10, 6))
plt.scatter(df['temp'], df['RH'], c=df['cluster'], cmap='viridis', marker='o', alpha=0.6)
plt.xlabel('Temperature')
plt.ylabel('Relative Humidity')
plt.title('K-means Clustering of Forest Fires Data')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()
