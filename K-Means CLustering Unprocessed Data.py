import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('k-mean-unprocessed-data.csv')

print("Duplicates:", df.duplicated().sum())

print("\nMissing Values:")
print(df.isna().sum())

cols = ['Annual_Income', 'Spending_Score_1_100']

df[cols] = df[cols].fillna(df[cols].median())

print("\nMissing Values:")
print(df.isna().sum())

features_name = [
    'Age',
    'Annual_Income',
    'Spending_Score_1_100'
]

features = df[features_name]

outliers_index = set()

for column in features_name:

    Q1 = features[column].quantile(0.25)
    Q3 = features[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = features[
        (features[column] < lower_bound) |
        (features[column] > upper_bound)
    ]

    outliers_index.update(outliers.index)

    print(
        f"Number of outliers in {column}: {len(outliers)}"
    )

print(
    f"\nTotal unique rows containing outliers: "
    f"{len(outliers_index)}"
)

features = df[
    ['Annual_Income', 'Spending_Score_1_100']
]

scaler = MinMaxScaler()

features_norm = scaler.fit_transform(features)

features_norm = pd.DataFrame(
    features_norm,
    columns=features.columns
)

plt.figure(figsize=(12, 9))

plt.scatter(
    features_norm['Annual_Income'],
    features_norm['Spending_Score_1_100'],
    s=20
)

plt.title('Normalized Customer Data')
plt.xlabel('Annual Income (Normalized)')
plt.ylabel('Spending Score (Normalized)')

plt.show()

X = features_norm.to_numpy()

wcss = []

for k in range(1, 11):

    kmeans = KMeans(
        n_clusters=k,
        init='k-means++',
        max_iter=300,
        n_init=10,
        random_state=0
    )

    kmeans.fit(X)

    wcss.append(kmeans.inertia_)

silhouette_coefficients = []

for k in range(2, 11):

    kmeans = KMeans(
        n_clusters=k,
        init='k-means++',
        max_iter=300,
        n_init=10,
        random_state=0
    )

    kmeans.fit(X)

    score = silhouette_score(
        X,
        kmeans.labels_
    )

    silhouette_coefficients.append(score)

plt.figure(figsize=(10, 8))

plt.subplot(2, 1, 1)

plt.plot(
    range(2, 11),
    silhouette_coefficients,
    marker='o'
)

plt.xticks(range(2, 11))

plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')

plt.title('Silhouette Method')

plt.subplot(2, 1, 2)

plt.plot(
    range(1, 11),
    wcss,
    marker='o'
)

plt.xticks(range(1, 11))

plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')

plt.title('Elbow Method')

plt.tight_layout()
plt.show()

best_k = range(2, 11)[
    np.argmax(silhouette_coefficients)
]

best_score = max(silhouette_coefficients)

print(
    f"\nBest number of clusters according to "
    f"Silhouette Score: {best_k}"
)

print(
    f"Best Silhouette Score: {best_score:.4f}"
)

kmeans = KMeans(
    n_clusters=best_k,
    init='k-means++',
    max_iter=500,
    n_init=10,
    random_state=0
)

kmeans_preds = kmeans.fit_predict(X)

df['Cluster'] = kmeans_preds

print("\nCustomers per Cluster:")
print(df['Cluster'].value_counts().sort_index())

plt.figure(figsize=(12, 9))

plt.scatter(
    X[:, 0],
    X[:, 1],
    c=kmeans_preds,
    s=20,
    cmap='tab10'
)

plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=200,
    c='black',
    marker='X',
    label='Centroids'
)

plt.title('Customer Segments using K-Means')

plt.xlabel('Annual Income (Normalized)')
plt.ylabel('Spending Score (Normalized)')

plt.legend()
plt.show()