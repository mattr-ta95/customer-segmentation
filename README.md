# Customer Segmentation Analysis

A comprehensive machine learning project that performs customer segmentation using clustering techniques on e-commerce data. This analysis helps businesses understand customer behavior patterns and optimize marketing strategies.

## Overview

This project analyzes customer data to identify distinct customer segments based on key behavioral and demographic features. The analysis uses unsupervised learning techniques including K-means clustering, hierarchical clustering, and dimensionality reduction methods (PCA and t-SNE) to uncover meaningful customer groups.

## Features

- **Customer Segmentation**: Identifies distinct customer groups using advanced clustering algorithms
- **Feature Engineering**: Creates meaningful features including frequency, recency, customer lifetime value (CLV), and customer age
- **Outlier Detection**: Uses Isolation Forest to identify and handle outliers
- **Dimensionality Reduction**: Applies PCA and t-SNE for data visualization
- **Comprehensive Analysis**: Provides detailed insights and business recommendations
- **Professional Visualizations**: Generates publication-ready charts and graphs

## Key Metrics Analyzed

- **Frequency**: Number of orders per customer (loyalty indicator)
- **Recency**: Days since last purchase (engagement indicator)
- **Customer Lifetime Value (CLV)**: Total revenue per customer (value indicator)
- **Average Unit Cost**: Average cost of items purchased (preference indicator)
- **Customer Age**: Age of customers (demographic indicator)

## Project Structure

```
Customer Segmentation/
├── data/
│   └── CUSTOMERS_CLEAN.csv          # Raw customer data
├── src/
│   └── customer_segmentation.py     # Main analysis script
├── results/                         # Generated outputs
│   ├── customer_segments.csv        # Final segmented data
│   ├── correlation_matrix.png       # Feature correlation heatmap
│   ├── feature_distributions.png    # Distribution plots
│   ├── elbow_method.png            # Elbow method for optimal k
│   ├── silhouette_method.png       # Silhouette analysis
│   ├── dendrogram.png              # Hierarchical clustering dendrogram
│   ├── cluster_analysis.png        # Cluster comparison boxplots
│   ├── pca_visualization.png       # PCA 2D visualization
│   └── tsne_visualization.png      # t-SNE 2D visualization
├── requirements.txt                 # Python dependencies
└── README.md                       # This file
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Customer\ Segmentation
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Ensure your data file is in the `data/` directory as `CUSTOMERS_CLEAN.csv`

2. Run the analysis:
```bash
python src/customer_segmentation.py
```

3. The script will:
   - Load and preprocess the data
   - Perform exploratory data analysis
   - Determine optimal number of clusters
   - Apply clustering algorithms
   - Generate visualizations
   - Provide business insights
   - Save results to the `results/` directory

## Data Requirements

The input data should be a CSV file with the following columns:
- `Customer ID`: Unique customer identifier
- `Order ID`: Unique order identifier
- `Order_Date`: Date of order (format: DDMMMYYYY)
- `Delivery_Date`: Date of delivery (format: DDMMMYYYY)
- `Total Revenue`: Revenue amount (currency format)
- `Unit Cost`: Cost per unit (currency format)
- `Customer_BirthDate`: Customer birth date (format: DDMMMYYYY)

## Methodology

### 1. Data Preprocessing
- Feature engineering to create segmentation variables
- Data aggregation (one row per customer)
- Outlier detection and removal using Isolation Forest

### 2. Exploratory Data Analysis
- Statistical summaries and correlation analysis
- Distribution analysis of key features
- Data quality assessment

### 3. Clustering Analysis
- **Elbow Method**: Determines optimal number of clusters based on within-cluster sum of squares
- **Silhouette Analysis**: Validates cluster quality and separation
- **Hierarchical Clustering**: Creates dendrogram for cluster relationship visualization
- **K-means Clustering**: Final segmentation using optimal k

### 4. Dimensionality Reduction
- **PCA**: Linear dimensionality reduction for visualization
- **t-SNE**: Non-linear dimensionality reduction for complex pattern visualization

### 5. Business Insights
- Cluster characterization and profiling
- Marketing recommendations for each segment
- Customer retention and acquisition strategies

## Results

The analysis generates several outputs:

- **Customer Segments**: CSV file with customer IDs and their assigned clusters
- **Visualizations**: Multiple charts showing data distributions, cluster characteristics, and 2D projections
- **Business Insights**: Detailed recommendations for each customer segment

## Key Insights

The analysis typically reveals customer segments such as:
- **High-Value Loyal Customers**: High frequency, high CLV, recent purchases
- **At-Risk Customers**: High recency (inactive), previously high value
- **New Customers**: Low frequency, recent first purchase
- **Price-Conscious Customers**: High frequency, low unit cost
- **Premium Customers**: High unit cost, moderate frequency

## Dependencies

- pandas >= 1.5.0
- numpy >= 1.21.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- scikit-learn >= 1.1.0
- scipy >= 1.9.0

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

**Matthew Russell**

## Acknowledgments

- Dataset provided for educational purposes
- Scikit-learn community for excellent machine learning tools
- Matplotlib and Seaborn for visualization capabilities
# customer-segmentation
