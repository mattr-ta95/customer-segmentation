#!/usr/bin/env python3
"""
Customer Segmentation Analysis

This script performs customer segmentation using clustering techniques on e-commerce data.
It analyzes customer behavior patterns including frequency, recency, customer lifetime value,
average unit cost, and customer age to identify distinct customer segments.

Author: Matthew Russell
Date: 2024
"""

import argparse
import logging
import os
import re
import sys
import warnings
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

# Random state for reproducibility
RANDOM_STATE = 42

# Outlier detection parameters
OUTLIER_CONTAMINATION = 0.05
OUTLIER_N_ESTIMATORS = 100

# Clustering parameters
MIN_CLUSTERS = 2
MAX_CLUSTERS = 8
KMEANS_N_INIT = 10

# Sampling parameters
DENDROGRAM_MAX_SAMPLES = 5000
DENDROGRAM_SAMPLE_FRAC = 0.3
TSNE_MAX_SAMPLES = 5000
TSNE_SAMPLE_FRAC = 0.2
TSNE_DEFAULT_PERPLEXITY = 30

# Visualization parameters
DPI = 300
FIGURE_SIZE_SMALL = (10, 6)
FIGURE_SIZE_MEDIUM = (10, 8)
FIGURE_SIZE_LARGE = (15, 10)
FIGURE_SIZE_XLARGE = (18, 12)

# Output paths
DEFAULT_INPUT_PATH = 'data/CUSTOMERS_CLEAN.csv'
DEFAULT_OUTPUT_DIR = 'results'
OUTPUT_CORRELATION = 'correlation_matrix.png'
OUTPUT_DISTRIBUTIONS = 'feature_distributions.png'
OUTPUT_ELBOW = 'elbow_method.png'
OUTPUT_SILHOUETTE = 'silhouette_method.png'
OUTPUT_DENDROGRAM = 'dendrogram.png'
OUTPUT_CLUSTER_ANALYSIS = 'cluster_analysis.png'
OUTPUT_PCA = 'pca_visualization.png'
OUTPUT_TSNE = 'tsne_visualization.png'
OUTPUT_CSV = 'customer_segments.csv'

# Required columns in input data
REQUIRED_COLUMNS = [
    'Customer ID',
    'Order ID',
    'Order_Date',
    'Delivery_Date',
    'Total Revenue',
    'Unit Cost',
    'Customer_BirthDate'
]

# Feature columns for clustering
CLUSTERING_FEATURES = ['Frequency', 'Recency', 'CLV', 'Average Unit Cost', 'Customer Age']

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

def setup_logging(verbose: bool = False) -> logging.Logger:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ============================================================================
# MATPLOTLIB CONFIGURATION
# ============================================================================

def setup_matplotlib(interactive: bool = True) -> None:
    """Setup matplotlib configuration."""
    if not interactive:
        matplotlib.use('Agg')  # Non-interactive backend

    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def currency_to_float(currency_str: str) -> float:
    """
    Convert currency string to float value.

    Args:
        currency_str: Currency string (e.g., "£1,234.56", "€1.234,56")

    Returns:
        Float value of the currency

    Raises:
        ValueError: If currency string cannot be converted
    """
    try:
        currency_str = str(currency_str).strip()

        # Remove currency symbols and whitespace
        currency_str = re.sub(r'[£€$\s]', '', currency_str)

        # Handle different decimal separators
        # If comma is used as decimal separator (European style)
        if ',' in currency_str and '.' not in currency_str:
            currency_str = currency_str.replace(',', '.')
        # If both exist, remove thousands separator
        elif ',' in currency_str and '.' in currency_str:
            # Assume comma is thousands separator (US/UK style)
            currency_str = currency_str.replace(',', '')

        # Remove any remaining non-numeric characters except decimal point and minus
        currency_str = re.sub(r'[^\d.-]', '', currency_str)

        return float(currency_str)
    except (ValueError, AttributeError) as e:
        logger.warning(f"Failed to convert currency '{currency_str}': {e}")
        return float('nan')

def validate_dataframe(df: pd.DataFrame, required_cols: list) -> None:
    """
    Validate that dataframe has required columns and is not empty.

    Args:
        df: DataFrame to validate
        required_cols: List of required column names

    Raises:
        ValueError: If validation fails
    """
    if df.empty:
        raise ValueError("Input dataframe is empty")

    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    logger.info(f"Dataframe validation passed. Shape: {df.shape}")

def ensure_output_dir(output_dir: str) -> None:
    """
    Ensure output directory exists.

    Args:
        output_dir: Path to output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.debug(f"Output directory ensured: {output_dir}")

def save_plot(filename: str, output_dir: str, show: bool = True) -> None:
    """
    Save plot to file and optionally show it.

    Args:
        filename: Name of output file
        output_dir: Directory to save to
        show: Whether to display plot interactively
    """
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=DPI, bbox_inches='tight')
    logger.info(f"Saved plot: {filepath}")

    if show:
        plt.show()
    else:
        plt.close()

def calculate_adaptive_sample_size(total_size: int, max_samples: int, frac: float) -> int:
    """
    Calculate adaptive sample size based on data size.

    Args:
        total_size: Total number of samples
        max_samples: Maximum number of samples to use
        frac: Fraction of data to sample if under max

    Returns:
        Number of samples to use
    """
    sample_size = min(max_samples, int(total_size * frac))
    return max(sample_size, min(100, total_size))  # At least 100 or total if less

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_and_preprocess_data(filepath: str) -> pd.DataFrame:
    """
    Load and preprocess the customer data.

    Args:
        filepath: Path to CSV file

    Returns:
        Aggregated DataFrame with one row per customer

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If data validation fails
    """
    logger.info(f"Loading data from: {filepath}")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        raise ValueError(f"Failed to read CSV file: {e}")

    logger.info(f"Data loaded. Shape: {df.shape}")

    # Validate required columns
    validate_dataframe(df, REQUIRED_COLUMNS)

    # Create a copy for processing
    processed_df = df.copy()

    # Feature Engineering
    logger.info("Performing feature engineering...")

    # Frequency: count of orders per customer
    processed_df['Frequency'] = processed_df.groupby('Customer ID')['Order ID'].transform('count')

    # Customer Lifetime Value (CLV): total revenue per customer
    # FIX: Work on processed_df directly, not original df
    processed_df['Total Revenue'] = processed_df['Total Revenue'].apply(currency_to_float)
    processed_df['CLV'] = processed_df['Total Revenue']

    # Average Unit Cost
    processed_df['Unit Cost'] = processed_df['Unit Cost'].apply(currency_to_float)
    processed_df['Average Unit Cost'] = processed_df['Unit Cost']

    # Date formatting and recency calculation
    try:
        processed_df['Order_Date'] = pd.to_datetime(processed_df['Order_Date'], format='%d%b%Y')
        processed_df['Delivery_Date'] = pd.to_datetime(processed_df['Delivery_Date'], format='%d%b%Y')
        processed_df['Birth'] = pd.to_datetime(processed_df['Customer_BirthDate'], format='%d%b%Y')
    except Exception as e:
        raise ValueError(f"Date parsing failed. Check date format (expected: DDMMMYYYY): {e}")

    today = pd.Timestamp('today')

    # Recency: days since last order
    processed_df['Recency'] = (today - processed_df['Order_Date']).dt.days

    # Customer Age
    processed_df['Customer Age'] = (today - processed_df['Birth']).dt.days // 365

    # Focus on key features for segmentation
    focus_features = ['Customer ID'] + CLUSTERING_FEATURES
    focus_df = processed_df[focus_features]

    # Aggregate to one customer per row
    # FIX: Use 'first' for Frequency since it's already computed per customer
    df_agg = focus_df.groupby('Customer ID').agg({
        'Frequency': 'first',  # Already computed correctly per customer
        'Recency': 'min',  # Most recent purchase date (minimum days)
        'CLV': 'sum',  # Total customer lifetime value
        'Average Unit Cost': 'median',  # Median to reduce influence of outliers
        'Customer Age': 'mean'  # Average age (should be same for all rows)
    }).reset_index()

    logger.info(f"Aggregated data shape: {df_agg.shape}")

    # Check for any infinite or extremely large values
    for col in CLUSTERING_FEATURES:
        if df_agg[col].isin([np.inf, -np.inf]).any():
            logger.warning(f"Infinite values found in {col}, replacing with NaN")
            df_agg[col] = df_agg[col].replace([np.inf, -np.inf], np.nan)

    return df_agg

# ============================================================================
# OUTLIER DETECTION
# ============================================================================

def detect_and_remove_outliers(df: pd.DataFrame, contamination: float = OUTLIER_CONTAMINATION) -> pd.DataFrame:
    """
    Detect and remove outliers using Isolation Forest.

    Args:
        df: Input DataFrame
        contamination: Expected proportion of outliers

    Returns:
        DataFrame with outliers removed
    """
    logger.info("Detecting outliers...")

    # Prepare data for outlier detection
    imputer = SimpleImputer(strategy='median')
    df_imputed = imputer.fit_transform(df[CLUSTERING_FEATURES])

    # Apply Isolation Forest
    iso_forest = IsolationForest(
        n_estimators=OUTLIER_N_ESTIMATORS,
        contamination=contamination,
        random_state=RANDOM_STATE
    )
    outlier_labels = iso_forest.fit_predict(df_imputed)

    # Filter out outliers
    df_clean = df[outlier_labels != -1].copy()

    n_outliers = len(df) - len(df_clean)
    logger.info(f"Removed {n_outliers} outliers ({n_outliers/len(df)*100:.1f}%)")
    logger.info(f"Data after outlier removal: {df_clean.shape}")

    return df_clean

# ============================================================================
# EXPLORATORY DATA ANALYSIS
# ============================================================================

def perform_eda(df: pd.DataFrame, output_dir: str, show_plots: bool = True) -> None:
    """
    Perform exploratory data analysis.

    Args:
        df: Input DataFrame
        output_dir: Directory to save plots
        show_plots: Whether to display plots interactively
    """
    logger.info("Performing exploratory data analysis...")

    # Basic statistics
    logger.info("\nData Summary:")
    logger.info(f"\n{df.describe()}")

    # Check for missing values
    missing = df.isna().sum()
    if missing.sum() > 0:
        logger.warning(f"\nMissing values:\n{missing[missing > 0]}")
    else:
        logger.info("No missing values found")

    # Correlation matrix
    plt.figure(figsize=FIGURE_SIZE_MEDIUM)
    corr_matrix = df[CLUSTERING_FEATURES].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title("Correlation Matrix of Customer Features")
    plt.tight_layout()
    save_plot(OUTPUT_CORRELATION, output_dir, show_plots)

    # Distribution plots
    fig, axes = plt.subplots(2, 3, figsize=FIGURE_SIZE_LARGE)
    axes = axes.ravel()

    for i, feature in enumerate(CLUSTERING_FEATURES):
        sns.histplot(df[feature], ax=axes[i], kde=True)
        axes[i].set_title(f'Distribution of {feature}')
        axes[i].set_xlabel(feature)

    # Remove empty subplot
    fig.delaxes(axes[5])
    plt.tight_layout()
    save_plot(OUTPUT_DISTRIBUTIONS, output_dir, show_plots)

# ============================================================================
# OPTIMAL CLUSTER DETERMINATION
# ============================================================================

def determine_optimal_clusters(
    df: pd.DataFrame,
    output_dir: str,
    show_plots: bool = True,
    n_clusters_override: Optional[int] = None
) -> Tuple[pd.DataFrame, int]:
    """
    Determine optimal number of clusters using Elbow and Silhouette methods.

    Args:
        df: Input DataFrame
        output_dir: Directory to save plots
        show_plots: Whether to display plots interactively
        n_clusters_override: Override automatic cluster selection

    Returns:
        Tuple of (scaled DataFrame, optimal number of clusters)
    """
    logger.info("Determining optimal number of clusters...")

    # Prepare data for clustering
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), CLUSTERING_FEATURES)
        ]
    )

    df_scaled = preprocessor.fit_transform(df)
    df_scaled = pd.DataFrame(df_scaled, columns=CLUSTERING_FEATURES)

    # If override is provided, use it
    if n_clusters_override:
        logger.info(f"Using user-specified number of clusters: {n_clusters_override}")
        return df_scaled, n_clusters_override

    # Elbow Method
    wcss = []
    k_range = range(1, MAX_CLUSTERS + 5)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=RANDOM_STATE, n_init=KMEANS_N_INIT)
        kmeans.fit(df_scaled)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=FIGURE_SIZE_SMALL)
    plt.plot(k_range, wcss, 'bo-')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.grid(True)
    save_plot(OUTPUT_ELBOW, output_dir, show_plots)

    # Silhouette Method with additional metrics
    silhouette_scores = []
    calinski_scores = []
    davies_bouldin_scores = []
    k_candidates = list(range(MIN_CLUSTERS, MAX_CLUSTERS + 1))

    for k in k_candidates:
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=RANDOM_STATE, n_init=KMEANS_N_INIT)
        cluster_labels = kmeans.fit_predict(df_scaled)

        silhouette_avg = silhouette_score(df_scaled, cluster_labels)
        calinski = calinski_harabasz_score(df_scaled, cluster_labels)
        davies_bouldin = davies_bouldin_score(df_scaled, cluster_labels)

        silhouette_scores.append(silhouette_avg)
        calinski_scores.append(calinski)
        davies_bouldin_scores.append(davies_bouldin)

        logger.info(f"k={k}: Silhouette={silhouette_avg:.3f}, "
                   f"Calinski-Harabasz={calinski:.1f}, Davies-Bouldin={davies_bouldin:.3f}")

    # Plot silhouette scores
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(k_candidates, silhouette_scores, 'ro-')
    axes[0].set_title('Silhouette Score')
    axes[0].set_xlabel('Number of Clusters (k)')
    axes[0].set_ylabel('Score (higher is better)')
    axes[0].grid(True)

    axes[1].plot(k_candidates, calinski_scores, 'go-')
    axes[1].set_title('Calinski-Harabasz Score')
    axes[1].set_xlabel('Number of Clusters (k)')
    axes[1].set_ylabel('Score (higher is better)')
    axes[1].grid(True)

    axes[2].plot(k_candidates, davies_bouldin_scores, 'bo-')
    axes[2].set_title('Davies-Bouldin Score')
    axes[2].set_xlabel('Number of Clusters (k)')
    axes[2].set_ylabel('Score (lower is better)')
    axes[2].grid(True)

    plt.tight_layout()
    save_plot(OUTPUT_SILHOUETTE, output_dir, show_plots)

    # Choose optimal k based on silhouette score
    optimal_k = k_candidates[np.argmax(silhouette_scores)]
    logger.info(f"Optimal number of clusters (by Silhouette): {optimal_k}")

    return df_scaled, optimal_k

# ============================================================================
# CLUSTERING ALGORITHMS
# ============================================================================

def perform_hierarchical_clustering(
    df_scaled: pd.DataFrame,
    output_dir: str,
    n_clusters: int = 5,
    show_plots: bool = True
) -> None:
    """
    Perform hierarchical clustering and create dendrogram.

    Args:
        df_scaled: Scaled DataFrame
        output_dir: Directory to save plots
        n_clusters: Number of clusters (for reference)
        show_plots: Whether to display plots interactively
    """
    logger.info("Performing hierarchical clustering...")

    # Use adaptive sampling for dendrogram to avoid memory issues
    sample_size = calculate_adaptive_sample_size(
        len(df_scaled),
        DENDROGRAM_MAX_SAMPLES,
        DENDROGRAM_SAMPLE_FRAC
    )

    if sample_size < len(df_scaled):
        df_sample = df_scaled.sample(n=sample_size, random_state=RANDOM_STATE)
        logger.info(f"Using {sample_size} samples for dendrogram (out of {len(df_scaled)})")
    else:
        df_sample = df_scaled
        logger.info(f"Using all {len(df_scaled)} samples for dendrogram")

    # Create linkage matrix and dendrogram
    Z = linkage(df_sample, method='ward')

    # Calculate optimal p for truncation
    p = min(15, sample_size // 10)

    plt.figure(figsize=(12, 8))
    dendrogram(Z, truncate_mode='lastp', p=p, leaf_rotation=90, leaf_font_size=12)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample Index or (Cluster Size)')
    plt.ylabel('Distance')
    save_plot(OUTPUT_DENDROGRAM, output_dir, show_plots)

def perform_kmeans_clustering(df_scaled: pd.DataFrame, n_clusters: int = 5) -> Tuple[np.ndarray, KMeans]:
    """
    Perform K-means clustering.

    Args:
        df_scaled: Scaled DataFrame
        n_clusters: Number of clusters

    Returns:
        Tuple of (cluster labels, KMeans model)
    """
    logger.info(f"Performing K-means clustering with {n_clusters} clusters...")

    kmeans = KMeans(
        n_clusters=n_clusters,
        init='k-means++',
        random_state=RANDOM_STATE,
        n_init=KMEANS_N_INIT
    )
    cluster_labels = kmeans.fit_predict(df_scaled)

    # Log cluster sizes
    unique, counts = np.unique(cluster_labels, return_counts=True)
    for cluster_id, count in zip(unique, counts):
        logger.info(f"Cluster {cluster_id}: {count} samples ({count/len(cluster_labels)*100:.1f}%)")

    return cluster_labels, kmeans

# ============================================================================
# CLUSTER ANALYSIS
# ============================================================================

def analyze_clusters(df: pd.DataFrame, cluster_labels: np.ndarray, output_dir: str, show_plots: bool = True) -> pd.DataFrame:
    """
    Analyze and visualize cluster characteristics.

    Args:
        df: Input DataFrame
        cluster_labels: Cluster assignments
        output_dir: Directory to save plots
        show_plots: Whether to display plots interactively

    Returns:
        DataFrame with cluster labels added
    """
    logger.info("Analyzing cluster characteristics...")

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

    logger.info("\nCluster Summary Statistics:")
    logger.info(f"\n{cluster_summary}")

    # Box plots for cluster analysis
    fig, axes = plt.subplots(2, 3, figsize=FIGURE_SIZE_XLARGE)
    axes = axes.ravel()

    for i, feature in enumerate(CLUSTERING_FEATURES):
        sns.boxplot(x='Cluster', y=feature, data=df_with_clusters, ax=axes[i])
        axes[i].set_title(f'{feature} by Cluster')
        axes[i].set_xlabel('Cluster')
        axes[i].set_ylabel(feature)

    # Remove empty subplot
    fig.delaxes(axes[5])
    plt.tight_layout()
    save_plot(OUTPUT_CLUSTER_ANALYSIS, output_dir, show_plots)

    return df_with_clusters

# ============================================================================
# DIMENSIONALITY REDUCTION
# ============================================================================

def perform_dimensionality_reduction(
    df_scaled: pd.DataFrame,
    cluster_labels: np.ndarray,
    output_dir: str,
    show_plots: bool = True
) -> None:
    """
    Perform PCA and t-SNE for visualization.

    Args:
        df_scaled: Scaled DataFrame
        cluster_labels: Cluster assignments
        output_dir: Directory to save plots
        show_plots: Whether to display plots interactively
    """
    logger.info("Performing dimensionality reduction...")

    # PCA
    pca = PCA(n_components=2)
    df_pca = pca.fit_transform(df_scaled)
    df_pca = pd.DataFrame(df_pca, columns=['PC1', 'PC2'])
    df_pca['Cluster'] = cluster_labels

    logger.info(f"PCA Explained Variance Ratio: {pca.explained_variance_ratio_}")
    logger.info(f"Total Explained Variance: {pca.explained_variance_ratio_.sum():.3f}")

    # PCA visualization
    plt.figure(figsize=FIGURE_SIZE_MEDIUM)
    sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=df_pca, palette='viridis', s=50, alpha=0.6)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.title('Customer Segments - PCA Visualization')
    plt.legend(title='Cluster')
    plt.grid(True, alpha=0.3)
    save_plot(OUTPUT_PCA, output_dir, show_plots)

    # t-SNE (on a sample to avoid memory issues)
    logger.info("Performing t-SNE...")

    sample_size = calculate_adaptive_sample_size(
        len(df_scaled),
        TSNE_MAX_SAMPLES,
        TSNE_SAMPLE_FRAC
    )

    if sample_size < len(df_scaled):
        sample_indices = np.random.choice(len(df_scaled), sample_size, replace=False)
        df_sample = df_scaled.iloc[sample_indices]
        sample_clusters = cluster_labels[sample_indices]
        logger.info(f"Using {sample_size} samples for t-SNE (out of {len(df_scaled)})")
    else:
        df_sample = df_scaled
        sample_clusters = cluster_labels
        logger.info(f"Using all {len(df_scaled)} samples for t-SNE")

    # Calculate appropriate perplexity
    # Rule of thumb: perplexity should be less than n_samples/3
    max_perplexity = sample_size // 3
    perplexity = min(TSNE_DEFAULT_PERPLEXITY, max_perplexity - 1)

    if perplexity < 5:
        logger.warning(f"Sample size too small for reliable t-SNE (perplexity={perplexity})")
        perplexity = max(5, perplexity)

    logger.info(f"Using perplexity={perplexity} for t-SNE")

    tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=perplexity)
    df_tsne = tsne.fit_transform(df_sample)
    df_tsne = pd.DataFrame(df_tsne, columns=['tSNE1', 'tSNE2'])
    df_tsne['Cluster'] = sample_clusters

    # t-SNE visualization
    plt.figure(figsize=FIGURE_SIZE_MEDIUM)
    sns.scatterplot(x='tSNE1', y='tSNE2', hue='Cluster', data=df_tsne, palette='viridis', s=50, alpha=0.6)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title('Customer Segments - t-SNE Visualization')
    plt.legend(title='Cluster')
    plt.grid(True, alpha=0.3)
    save_plot(OUTPUT_TSNE, output_dir, show_plots)

# ============================================================================
# BUSINESS INSIGHTS
# ============================================================================

def generate_insights(df_with_clusters: pd.DataFrame) -> None:
    """
    Generate business insights from the clustering results.

    Args:
        df_with_clusters: DataFrame with cluster labels
    """
    logger.info("\n" + "="*70)
    logger.info("BUSINESS INSIGHTS AND RECOMMENDATIONS")
    logger.info("="*70)

    cluster_summary = df_with_clusters.groupby('Cluster').agg({
        'Frequency': 'mean',
        'Recency': 'mean',
        'CLV': 'mean',
        'Average Unit Cost': 'mean',
        'Customer Age': 'mean',
        'Customer ID': 'count'
    }).round(2)

    cluster_summary.columns = ['Avg_Frequency', 'Avg_Recency', 'Avg_CLV', 'Avg_Unit_Cost', 'Avg_Age', 'Count']

    logger.info("\nCluster Characteristics:")
    logger.info(f"\n{cluster_summary}")

    # Calculate global means for comparison
    global_freq = df_with_clusters['Frequency'].mean()
    global_recency = df_with_clusters['Recency'].mean()
    global_clv = df_with_clusters['CLV'].mean()

    logger.info("\nKey Insights:")
    for cluster in sorted(df_with_clusters['Cluster'].unique()):
        cluster_data = df_with_clusters[df_with_clusters['Cluster'] == cluster]
        avg_freq = cluster_data['Frequency'].mean()
        avg_recency = cluster_data['Recency'].mean()
        avg_clv = cluster_data['CLV'].mean()
        avg_cost = cluster_data['Average Unit Cost'].mean()
        count = len(cluster_data)

        logger.info(f"\nCluster {cluster} ({count} customers, {count/len(df_with_clusters)*100:.1f}%):")
        logger.info(f"  - Average Frequency: {avg_freq:.1f} orders")
        logger.info(f"  - Average Recency: {avg_recency:.0f} days since last order")
        logger.info(f"  - Average CLV: ${avg_clv:.2f}")
        logger.info(f"  - Average Unit Cost: ${avg_cost:.2f}")

        # Business recommendations
        recommendations = []
        if avg_freq > global_freq * 1.2:
            recommendations.append("High-value customers - focus on retention and loyalty programs")
        if avg_recency > global_recency * 1.5:
            recommendations.append("At-risk customers - re-engagement campaigns needed")
        if avg_clv > global_clv * 1.3:
            recommendations.append("High CLV segment - premium service offerings and VIP treatment")
        if avg_freq < global_freq * 0.5:
            recommendations.append("Low engagement - consider targeted promotions or surveys")

        if recommendations:
            logger.info("  RECOMMENDATIONS:")
            for rec in recommendations:
                logger.info(f"    • {rec}")

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main() -> None:
    """Main function to run the customer segmentation analysis."""
    parser = argparse.ArgumentParser(
        description='Customer Segmentation Analysis using Clustering',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s
  %(prog)s --input data/customers.csv --output results/
  %(prog)s --n-clusters 5 --no-interactive
  %(prog)s --verbose --no-interactive
        """
    )

    parser.add_argument(
        '--input',
        type=str,
        default=DEFAULT_INPUT_PATH,
        help=f'Input CSV file path (default: {DEFAULT_INPUT_PATH})'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f'Output directory for results (default: {DEFAULT_OUTPUT_DIR})'
    )

    parser.add_argument(
        '--n-clusters',
        type=int,
        default=None,
        help='Number of clusters (if not specified, optimal k will be determined automatically)'
    )

    parser.add_argument(
        '--no-interactive',
        action='store_true',
        help='Run in batch mode without showing plots interactively'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    parser.add_argument(
        '--contamination',
        type=float,
        default=OUTLIER_CONTAMINATION,
        help=f'Expected proportion of outliers (default: {OUTLIER_CONTAMINATION})'
    )

    args = parser.parse_args()

    # Setup logging
    global logger
    logger = setup_logging(args.verbose)

    # Setup matplotlib
    setup_matplotlib(not args.no_interactive)

    # Ensure output directory exists
    ensure_output_dir(args.output_dir)

    logger.info("="*70)
    logger.info("Customer Segmentation Analysis")
    logger.info("="*70)
    logger.info(f"Input file: {args.input}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Interactive mode: {not args.no_interactive}")

    try:
        # Load and preprocess data
        df_agg = load_and_preprocess_data(args.input)

        # Remove outliers
        df_clean = detect_and_remove_outliers(df_agg, args.contamination)

        # Perform EDA
        perform_eda(df_clean, args.output_dir, not args.no_interactive)

        # Determine optimal clusters
        df_scaled, optimal_k = determine_optimal_clusters(
            df_clean,
            args.output_dir,
            not args.no_interactive,
            args.n_clusters
        )

        # Perform hierarchical clustering
        perform_hierarchical_clustering(df_scaled, args.output_dir, optimal_k, not args.no_interactive)

        # Perform K-means clustering
        cluster_labels, kmeans_model = perform_kmeans_clustering(df_scaled, optimal_k)

        # Analyze clusters
        df_with_clusters = analyze_clusters(df_clean, cluster_labels, args.output_dir, not args.no_interactive)

        # Dimensionality reduction and visualization
        perform_dimensionality_reduction(df_scaled, cluster_labels, args.output_dir, not args.no_interactive)

        # Generate insights
        generate_insights(df_with_clusters)

        # Save results
        output_csv = os.path.join(args.output_dir, OUTPUT_CSV)
        df_with_clusters.to_csv(output_csv, index=False)
        logger.info(f"\nResults saved to '{output_csv}'")

        logger.info("\n" + "="*70)
        logger.info("Analysis complete!")
        logger.info("="*70)

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        logger.error(f"\nPlease ensure your data file exists at: {args.input}")
        logger.error("See data/README.md for data format requirements")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Data validation error: {e}")
        logger.error("\nPlease check your data format. See data/README.md for requirements")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
