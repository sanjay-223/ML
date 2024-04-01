import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_csv("../URL_data.csv")

X = df.drop('target', axis=1)


kmeans = KMeans(n_clusters=2, random_state=42) 
kmeans.fit(X)

cluster_labels = kmeans.labels_


silhouette_score_val = silhouette_score(X, cluster_labels)

print("Silhouette Score:", silhouette_score_val)
