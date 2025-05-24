import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

# Membuat direktori untuk menyimpan visualisasi
if not os.path.exists('static'):
    os.makedirs('static')

# Load dataset
print("1. Load Dataset")
df = pd.read_csv("Ecommerce_Consumer.csv")
print("\nContoh data:")
print(df.head())

# Pilih fitur yang akan digunakan
print("\n2. Preprocessing Data")
features = ["Age", "Purchase_Amount", "Frequency_of_Purchase", 
           "Product_Rating", "Customer_Satisfaction"]
X = df[features]

print("\nJumlah data sebelum preprocessing:", len(X))

# Cek dan handle missing values
print("\nJumlah missing values:")
print(X.isnull().sum())
X = X.fillna(X.mean())

# Cek duplikat
duplicates = X.duplicated().sum()
print(f"\nJumlah data duplikat: {duplicates}")
X = X.drop_duplicates()
print(f"Jumlah data setelah menghapus duplikat: {len(X)}")

# Normalisasi data
print("\nMelakukan normalisasi data...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Menentukan jumlah cluster dengan metode Elbow
print("\n3. Menentukan jumlah cluster optimal dengan metode Elbow")
inertia = []
k_range = range(1, 10)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot hasil Elbow Method
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia, marker='o')
plt.xlabel('Jumlah Cluster (K)')
plt.ylabel('Inertia')
plt.title('Metode Elbow untuk Menentukan K Optimal')
plt.savefig('static/elbow_plot.png')
plt.close()

# Menggunakan K optimal (K=3)
print("\n4. Implementasi K-Means dengan K=3")
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Menampilkan hasil cluster
print("\nHasil Clustering Pelanggan E-commerce:")
print(df[['Customer_ID', 'Age', 'Purchase_Amount', 'Frequency_of_Purchase',
          'Product_Rating', 'Customer_Satisfaction', 'Cluster']].head(10))

# Visualisasi hasil clustering
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Purchase_Amount', y='Frequency_of_Purchase', 
                hue='Cluster', palette='viridis')
plt.title('Hasil Clustering Pelanggan E-commerce')
plt.xlabel('Jumlah Pembelian ($)')
plt.ylabel('Frekuensi Pembelian')
plt.savefig('static/cluster_plot.png')
plt.close()

# Analisis statistik setiap cluster
print("\n5. Analisis Hasil Clustering")
cluster_stats = df.groupby('Cluster').agg({
    'Age': ['mean', 'std'],
    'Purchase_Amount': ['mean', 'std'],
    'Frequency_of_Purchase': ['mean', 'std'],
    'Product_Rating': ['mean', 'std'],
    'Customer_Satisfaction': ['mean', 'std']
}).round(2)

print("\nStatistik per Cluster:")
print(cluster_stats)

# Interpretasi hasil
print("\n6. Interpretasi Hasil")
print("\nKarakteristik Cluster:")
for cluster in range(3):
    stats = df[df['Cluster'] == cluster].describe()
    print(f"\nCluster {cluster}:")
    print(f"- Rata-rata Usia: {stats['Age']['mean']:.2f}")
    print(f"- Rata-rata Pembelian: ${stats['Purchase_Amount']['mean']:.2f}")
    print(f"- Rata-rata Frekuensi: {stats['Frequency_of_Purchase']['mean']:.2f} kali/bulan")
    print(f"- Rata-rata Rating: {stats['Product_Rating']['mean']:.2f}")
    print(f"- Rata-rata Kepuasan: {stats['Customer_Satisfaction']['mean']:.2f}")

# Simpan hasil untuk web app
df.to_csv('static/clustering_results.csv', index=False)
cluster_stats.to_csv('static/cluster_stats.csv')

print("\nAnalisis selesai. Hasil telah disimpan di folder 'static'")
