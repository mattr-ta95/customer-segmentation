#!/usr/bin/env python3
"""
Customer Segmentation Analysis

This script performs customer segmentation using clustering techniques on e-commerce data.
It analyzes customer behavior patterns including frequency, recency, customer lifetime value,
average unit cost, and customer age to identify distinct customer segments.

Author: Matthew Russell
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from scipy.cluster.hierarchy import linkage, dendrogram
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def currency_to_float(currency_str):
    """Convert currency string to float value."""
    try:
        currency_str = str(currency_str)
        currency_str = currency_str.strip()
        # Remove currency symbols and thousands separators
        currency_str = re.sub(r'[^\d.,€£-]', '', currency_str)
        # Replace commas with periods if they are used as decimal separators
        if ',' in currency_str and '.' not in currency_str:
            currency_str = currency_str.replace(',', '.')
        return float(currency_str)
    except ValueError:
        return float('nan')

def load_and_preprocess_data(filepath):
    """Load and preprocess the customer data."""
    print("Loading data...")
    df = pd.read_csv(filepath)
    print(f"Data shape: {df.shape}")
    
    # Create a copy for processing
    processed_df = df.copy()
    
    # Feature Engineering
    print("Performing feature engineering...")
    
    # Frequency: count of orders per customer
    processed_df['Frequency'] = df.groupby('Customer ID')['Order ID'].transform('count')
    
    # Customer Lifetime Value (CLV): total revenue per customer
    df['Total Revenue'] = df['Total Revenue'].apply(currency_to_float)
    processed_df['Total Revenue'] = pd.to_numeric(df['Total Revenue'])
    processed_df['CLV'] = processed_df['Total Revenue']
    
    # Average Unit Cost
    processed_df['Unit Cost'] = processed_df['Unit Cost'].apply(currency_to_float)
    processed_df['Average Unit Cost'] = processed_df['Unit Cost']
    
    # Date formatting and recency calculation
    processed_df['Order_Date'] = pd.to_datetime(processed_df['Order_Date'], format='%d%b%Y')
    processed_df['Delivery_Date'] = pd.to_datetime(processed_df['Delivery_Date'], format='%d%b%Y')
    today = pd.Timestamp('today')
    
    # Recency: days since last order
    processed_df['Recency'] = (today - processed_df['Order_Date']).dt.days
    
    # Customer Age
    processed_df['Birth'] = pd.to_datetime(df['Customer_BirthDate'], format='%d%b%Y')
    processed_df['Customer Age'] = (today - processed_df['Birth']).dt.days // 365
    
    # Focus on key features for segmentation
    focus_features = ['Customer ID', 'Frequency', 'Recency', 'CLV', 'Average Unit Cost', 'Customer Age']
    focus_df = processed_df[focus_features]
    
    # Aggregate to one customer per row
    df_agg = focus_df.groupby('Customer ID').agg({
        'Frequency': 'sum',
        'Recency': 'min',  # most recent purchase date
        'CLV': 'sum',
        'Average Unit Cost': 'median',  # to reduce influence of outliers
        'Customer Age': 'mean'
    }).reset_index()
    
    print(f"Aggregated data shape: {df_agg.shape}")
    return df_agg

def detect_and_remove_outliers(df):
    """Detect and remove outliers using Isolation Forest."""
    print("Detecting outliers...")
    
    # Prepare data for outlier detection
    features = ['Frequency', 'Recency', 'CLV', 'Average Unit Cost', 'Customer Age']
    imputer = SimpleImputer(strategy='median')
    df_imputed = imputer.fit_transform(df[features])
    
    # Apply Isolation Forest
    iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=10)
    outlier_labels = iso_forest.fit_predict(df_imputed)
    
    # Filter out outliers
    df_clean = df[outlier_labels != -1].copy()
    
    print(f"Data after outlier removal: {df_clean.shape}")
    return df_clean

def perform_eda(df):
    """Perform exploratory data analysis."""
    print("Performing exploratory data analysis...")
    
    # Basic statistics
    print("\nData Summary:")
    print(df.describe())
    
    # Check for missing values
    print(f"\nMissing values:\n{df.isna().sum()}")
    
    # Correlation matrix
    plt.figure(figsize=(10, 8))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title("Correlation Matrix of Customer Features")
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Distribution plots
    features = ['Frequency', 'Recency', 'CLV', 'Average Unit Cost', 'Customer Age']
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, feature in enumerate(features):
        sns.histplot(df[feature], ax=axes[i], kde=True)
        axes[i].set_title(f'Distribution of {feature}')
        axes[i].set_xlabel(feature)
    
    # Remove empty subplot
    fig.delaxes(axes[5])
    plt.tight_layout()
    plt.savefig('feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()

def determine_optimal_clusters(df):
    """Determine optimal number of clusters using Elbow and Silhouette methods."""
    print("Determining optimal number of clusters...")
    
    # Prepare data for clustering
    features = ['Frequency', 'Recency', 'CLV', 'Average Unit Cost', 'Customer Age']
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), features)
        ]
    )
    
    df_scaled = preprocessor.fit_transform(df)
    df_scaled = pd.DataFrame(df_scaled, columns=features)
    
    # Elbow Method
    wcss = []
    k_range = range(1, 13)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=50, n_init=10)
        kmeans.fit(df_scaled)
        wcss.append(kmeans.inertia_)
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, wcss, 'bo-')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.grid(True)
    plt.savefig('elbow_method.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Silhouette Method
    silhouette_scores = []
    k_candidates = [2, 3, 4, 5, 6, 7, 8]
    
    for k in k_candidates:
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=100, n_init=10)
        cluster_labels = kmeans.fit_predict(df_scaled)
        silhouette_avg = silhouette_score(df_scaled, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        print(f"k={k}: Silhouette Score = {silhouette_avg:.3f}")
    
    # Plot silhouette scores
    plt.figure(figsize=(10, 6))
    plt.plot(k_candidates, silhouette_scores, 'ro-')
    plt.title('Silhouette Method for Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    plt.savefig('silhouette_method.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Choose optimal k (you can modify this logic)
    optimal_k = k_candidates[np.argmax(silhouette_scores)]
    print(f"Optimal number of clusters: {optimal_k}")
    
    return df_scaled, optimal_k

def perform_hierarchical_clustering(df_scaled, n_clusters=5):
    """Perform hierarchical clustering and create dendrogram."""
    print("Performing hierarchical clustering...")
    
    # Use a sample for dendrogram to avoid memory issues
    df_sample = df_scaled.sample(frac=0.3, random_state=1000)
    
    # Create linkage matrix and dendrogram
    Z = linkage(df_sample, method='ward')
    
    plt.figure(figsize=(12, 8))
    dendrogram(Z, truncate_mode='lastp', p=15, leaf_rotation=90, leaf_font_size=12)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample Index or (Cluster Size)')
    plt.ylabel('Distance')
    plt.savefig('dendrogram.png', dpi=300, bbox_inches='tight')
    plt.show()

def perform_kmeans_clustering(df_scaled, n_clusters=5):
    """Perform K-means clustering."""
    print(f"Performing K-means clustering with {n_clusters} clusters...")
    
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=100, n_init=10)
    cluster_labels = kmeans.fit_predict(df_scaled)
    
    return cluster_labels, kmeans

def analyze_clusters(df, cluster_labels):
    """Analyze and visualize cluster characteristics."""
    print("Analyzing cluster characteristics...")
    
    # Add cluster labels to dataframe
    df_with_clusters = df.copy()
    df_with_clusters['Cluster'] = cluster_labels
    
    # Cluster summary statistics
    cluster_summary = df_with_clusters.groupby('Cluster').agg({
        'Frequency': ['mean', 'std'],
        'Recency': ['mean', 'std'],
        'CLV': ['mean', 'std'],
        'Average Unit Cost': ['mean', 'std'],
        'Customer Age': ['mean', 'std']
    }).round(2)
    
    print("\nCluster Summary Statistics:")
    print(cluster_summary)
    
    # Box plots for cluster analysis
    features = ['Frequency', 'Recency', 'CLV', 'Average Unit Cost', 'Customer Age']
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    for i, feature in enumerate(features):
        sns.boxplot(x='Cluster', y=feature, data=df_with_clusters, ax=axes[i])
        axes[i].set_title(f'{feature} by Cluster')
        axes[i].set_xlabel('Cluster')
        axes[i].set_ylabel(feature)
    
    # Remove empty subplot
    fig.delaxes(axes[5])
    plt.tight_layout()
    plt.savefig('cluster_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df_with_clusters

def perform_dimensionality_reduction(df_scaled, cluster_labels):
    """Perform PCA and t-SNE for visualization."""
    print("Performing dimensionality reduction...")
    
    # PCA
    pca = PCA(n_components=2)
    df_pca = pca.fit_transform(df_scaled)
    df_pca = pd.DataFrame(df_pca, columns=['PC1', 'PC2'])
    df_pca['Cluster'] = cluster_labels
    
    print(f"PCA Explained Variance Ratio: {pca.explained_variance_ratio_}")
    print(f"Total Explained Variance: {pca.explained_variance_ratio_.sum():.3f}")
    
    # PCA visualization
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=df_pca, palette='viridis', s=50)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.title('Customer Segments - PCA Visualization')
    plt.legend(title='Cluster')
    plt.grid(True, alpha=0.3)
    plt.savefig('pca_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # t-SNE (on a sample to avoid memory issues)
    print("Performing t-SNE...")
    df_sample = df_scaled.sample(frac=0.2, random_state=1000)
    tsne = TSNE(n_components=2, random_state=100, perplexity=30)
    df_tsne = tsne.fit_transform(df_sample)
    df_tsne = pd.DataFrame(df_tsne, columns=['tSNE1', 'tSNE2'])
    
    # Get cluster labels for the sample
    sample_indices = df_sample.index
    sample_clusters = cluster_labels[sample_indices]
    df_tsne['Cluster'] = sample_clusters
    
    # t-SNE visualization
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='tSNE1', y='tSNE2', hue='Cluster', data=df_tsne, palette='viridis', s=50)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title('Customer Segments - t-SNE Visualization')
    plt.legend(title='Cluster')
    plt.grid(True, alpha=0.3)
    plt.savefig('tsne_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_insights(df_with_clusters):
    """Generate business insights from the clustering results."""
    print("\n" + "="*50)
    print("BUSINESS INSIGHTS AND RECOMMENDATIONS")
    print("="*50)
    
    cluster_summary = df_with_clusters.groupby('Cluster').agg({
        'Frequency': 'mean',
        'Recency': 'mean',
        'CLV': 'mean',
        'Average Unit Cost': 'mean',
        'Customer Age': 'mean',
        'Customer ID': 'count'
    }).round(2)
    
    cluster_summary.columns = ['Avg_Frequency', 'Avg_Recency', 'Avg_CLV', 'Avg_Unit_Cost', 'Avg_Age', 'Count']
    
    print("\nCluster Characteristics:")
    print(cluster_summary)
    
    print("\nKey Insights:")
    for cluster in sorted(df_with_clusters['Cluster'].unique()):
        cluster_data = df_with_clusters[df_with_clusters['Cluster'] == cluster]
        avg_freq = cluster_data['Frequency'].mean()
        avg_recency = cluster_data['Recency'].mean()
        avg_clv = cluster_data['CLV'].mean()
        avg_cost = cluster_data['Average Unit Cost'].mean()
        count = len(cluster_data)
        
        print(f"\nCluster {cluster} ({count} customers):")
        print(f"  - Average Frequency: {avg_freq:.1f} orders")
        print(f"  - Average Recency: {avg_recency:.0f} days since last order")
        print(f"  - Average CLV: ${avg_clv:.2f}")
        print(f"  - Average Unit Cost: ${avg_cost:.2f}")
        
        # Business recommendations
        if avg_freq > df_with_clusters['Frequency'].mean():
            print(f"  - RECOMMENDATION: High-value customers - focus on retention")
        if avg_recency > df_with_clusters['Recency'].mean():
            print(f"  - RECOMMENDATION: At-risk customers - re-engagement campaigns needed")
        if avg_clv > df_with_clusters['CLV'].mean():
            print(f"  - RECOMMENDATION: High CLV segment - premium service offerings")

def main():
    """Main function to run the customer segmentation analysis."""
    print("Customer Segmentation Analysis")
    print("="*40)
    
    # Load and preprocess data
    df_agg = load_and_preprocess_data('data/CUSTOMERS_CLEAN.csv')
    
    # Remove outliers
    df_clean = detect_and_remove_outliers(df_agg)
    
    # Perform EDA
    perform_eda(df_clean)
    
    # Determine optimal clusters
    df_scaled, optimal_k = determine_optimal_clusters(df_clean)
    
    # Perform hierarchical clustering
    perform_hierarchical_clustering(df_scaled, optimal_k)
    
    # Perform K-means clustering
    cluster_labels, kmeans_model = perform_kmeans_clustering(df_scaled, optimal_k)
    
    # Analyze clusters
    df_with_clusters = analyze_clusters(df_clean, cluster_labels)
    
    # Dimensionality reduction and visualization
    perform_dimensionality_reduction(df_scaled, cluster_labels)
    
    # Generate insights
    generate_insights(df_with_clusters)
    
    # Save results
    df_with_clusters.to_csv('results/customer_segments.csv', index=False)
    print(f"\nResults saved to 'results/customer_segments.csv'")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
