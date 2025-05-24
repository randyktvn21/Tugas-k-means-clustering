import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file, flash
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-here')  # untuk flash messages

def preprocess_data(data):
    
    # Handle missing values
    data = data.fillna(data.mean())
    
    # Remove duplicates
    data = data.drop_duplicates()
    
    # Normalize data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    return data_scaled

def perform_clustering(X, n_clusters=3):
    """
    Melakukan clustering menggunakan K-Means
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)
    return clusters

def create_cluster_plot(df, clusters):
    """
    Membuat visualisasi hasil clustering
    """
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(df['Purchase_Amount'], df['Frequency_of_Purchase'], 
                         c=clusters, cmap='viridis')
    plt.xlabel('Jumlah Pembelian ($)')
    plt.ylabel('Frekuensi Pembelian')
    plt.title('Segmentasi Pelanggan E-commerce')
    plt.colorbar(scatter)
    
    img = BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return plot_url

def create_elbow_plot(X_scaled):
    """
    Membuat dan menyimpan plot metode Elbow
    """
    inertias = []
    K = range(1, 10)
    
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
    
    plt.figure(figsize=(10, 6))
    plt.plot(K, inertias, 'bx-')
    plt.xlabel('Jumlah Cluster (K)')
    plt.ylabel('Inertia')
    plt.title('Metode Elbow untuk Penentuan Jumlah Cluster Optimal')
    plt.grid(True)
    plt.savefig('static/elbow_plot.png')
    plt.close()

@app.route('/')
def home():
    # Load saved results if available
    try:
        stats = pd.read_csv('static/cluster_stats.csv')
        samples = pd.read_csv('static/sample_customers.csv')
        has_saved_results = True
    except:
        has_saved_results = False
        stats = None
        samples = None
    
    return render_template('index.html', 
                         has_saved_results=has_saved_results,
                         cluster_stats=stats.to_html(classes='table table-striped') if stats is not None else None,
                         sample_customers=samples.to_html(classes='table table-striped') if samples is not None else None,
                         prediction_result=request.args.get('prediction_result'))

@app.route('/predict_cluster', methods=['POST'])
def predict_cluster():
    try:
        # Get input data
        new_data = {
            'Age': float(request.form['age']),
            'Purchase_Amount': float(request.form['purchase_amount']),
            'Frequency_of_Purchase': float(request.form['frequency']),
            'Product_Rating': float(request.form['rating']),
            'Customer_Satisfaction': float(request.form['satisfaction'])
        }
        
        # Create DataFrame
        new_df = pd.DataFrame([new_data])
        
        # Load existing data
        df = pd.read_csv('Ecommerce_Consumer.csv')
        features = ["Age", "Purchase_Amount", "Frequency_of_Purchase", 
                   "Product_Rating", "Customer_Satisfaction"]
        
        # Combine new data with existing data for proper scaling
        combined_df = pd.concat([df[features], new_df], ignore_index=True)
        
        # Preprocess data
        X_scaled = preprocess_data(combined_df)
        
        # Generate Elbow plot
        create_elbow_plot(X_scaled)
        
        # Perform clustering
        clusters = perform_clustering(X_scaled)
        
        # Create clustering visualization
        plt.figure(figsize=(10, 6))
        plt.scatter(combined_df['Purchase_Amount'], combined_df['Frequency_of_Purchase'], 
                   c=clusters, cmap='viridis', alpha=0.6)
        
        # Highlight the new data point
        plt.scatter(new_data['Purchase_Amount'], new_data['Frequency_of_Purchase'],
                   color='red', marker='*', s=200, label='Data Anda')
        
        plt.xlabel('Jumlah Pembelian ($)')
        plt.ylabel('Frekuensi Pembelian')
        plt.title('Hasil Clustering dengan Posisi Data Anda')
        plt.legend()
        plt.savefig('static/clustering_result.png')
        plt.close()
        
        # Get cluster for new data (last row)
        new_cluster = clusters[-1]
        
        # Calculate average purchase amount for each cluster
        df['Cluster'] = clusters[:-1]  # Exclude the new data point
        cluster_means = df.groupby('Cluster')['Purchase_Amount'].agg(['min', 'max', 'mean'])
        
        # Sort clusters by mean purchase amount
        sorted_clusters = cluster_means.sort_values('mean')
        cluster_mapping = {old: new for new, old in enumerate(sorted_clusters.index)}
        
        # Map the new cluster to the correct category
        mapped_cluster = cluster_mapping[new_cluster]
        
        # Define cluster characteristics based on actual data ranges
        characteristics = {
            0: f"Pelanggan Ekonomis (Pembelian ${sorted_clusters.iloc[0]['min']:.0f}-${sorted_clusters.iloc[0]['max']:.0f} per transaksi)",
            1: f"Pelanggan Regular (Pembelian ${sorted_clusters.iloc[1]['min']:.0f}-${sorted_clusters.iloc[1]['max']:.0f} per transaksi)",
            2: f"Pelanggan Premium (Pembelian ${sorted_clusters.iloc[2]['min']:.0f}-${sorted_clusters.iloc[2]['max']:.0f} per transaksi)"
        }
        
        result_message = f"Pelanggan termasuk dalam Cluster {mapped_cluster}: {characteristics[mapped_cluster]}"
        
        return render_template('index.html', 
                             prediction_result=result_message,
                             show_visualizations=True,
                             has_saved_results=False)
    
    except Exception as e:
        return render_template('index.html', 
                             prediction_result=f"Error: {str(e)}",
                             show_visualizations=False,
                             has_saved_results=False)

@app.route('/cluster', methods=['POST'])
def cluster():
    try:
        n_clusters = int(request.form.get('n_clusters', 3))
        
        # 1. Load dataset
        df = pd.read_csv('Ecommerce_Consumer.csv')
        features = ["Age", "Purchase_Amount", "Frequency_of_Purchase", 
                   "Product_Rating", "Customer_Satisfaction"]
        X = df[features]
        
        # 2. Preprocessing
        X_scaled = preprocess_data(X)
        
        # 3. Clustering
        clusters = perform_clustering(X_scaled, n_clusters)
        
        # 4. Visualisasi
        cluster_plot = create_cluster_plot(df, clusters)
        
        # 5. Analisis hasil
        df['Cluster'] = clusters
        cluster_stats = df.groupby('Cluster').agg({
            'Age': ['mean', 'std'],
            'Purchase_Amount': ['mean', 'std'],
            'Frequency_of_Purchase': ['mean', 'std'],
            'Product_Rating': ['mean', 'std'],
            'Customer_Satisfaction': ['mean', 'std']
        }).round(2)
        
        # Get sample customers
        sample_customers = df.groupby('Cluster').apply(
            lambda x: x[['Customer_ID', 'Age', 'Purchase_Amount', 'Frequency_of_Purchase']].head(3)
        ).reset_index(drop=True)
        
        # Save results
        df.to_csv('static/clustering_results.csv', index=False)
        cluster_stats.to_csv('static/cluster_stats.csv')
        sample_customers.to_csv('static/sample_customers.csv', index=False)
        
        return render_template('results.html',
                             cluster_plot=cluster_plot,
                             cluster_stats=cluster_stats.to_html(classes='table table-striped'),
                             sample_customers=sample_customers.to_html(classes='table table-striped'),
                             n_clusters=n_clusters)
    
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/download_results', methods=['GET'])
def download_results():
    try:
        if os.path.exists('static/clustering_results.csv'):
            return send_file(
                'static/clustering_results.csv',
                mimetype='text/csv',
                as_attachment=True,
                download_name='clustering_results.csv'
            )
        else:
            return jsonify({'error': 'No results available'})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 