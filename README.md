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
customer-segmentation/
├── data/
│   ├── .gitkeep                     # Keeps directory in git
│   └── README.md                    # Data format documentation
├── src/
│   ├── __init__.py                  # Package initialization
│   └── customer_segmentation.py     # Main analysis script
├── tests/
│   ├── __init__.py
│   └── test_customer_segmentation.py # Unit tests
├── results/                         # Generated outputs (git-ignored)
│   ├── customer_segments.csv        # Final segmented data
│   ├── correlation_matrix.png       # Feature correlation heatmap
│   ├── feature_distributions.png    # Distribution plots
│   ├── elbow_method.png            # Elbow method for optimal k
│   ├── silhouette_method.png       # Silhouette + validation metrics
│   ├── dendrogram.png              # Hierarchical clustering dendrogram
│   ├── cluster_analysis.png        # Cluster comparison boxplots
│   ├── pca_visualization.png       # PCA 2D visualization
│   └── tsne_visualization.png      # t-SNE 2D visualization
├── .gitignore                       # Git ignore rules
├── config.yaml                      # Configuration file
├── LICENSE                          # MIT License
├── requirements.txt                 # Python dependencies
└── README.md                       # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/mattr-ta95/customer-segmentation.git
cd customer-segmentation
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

### Basic Usage

1. Ensure your data file is in the `data/` directory as `CUSTOMERS_CLEAN.csv`

2. Run the analysis with default settings:
```bash
python src/customer_segmentation.py
```

### Advanced Usage

**Run with custom input/output paths:**
```bash
python src/customer_segmentation.py --input data/my_data.csv --output-dir results/
```

**Specify number of clusters:**
```bash
python src/customer_segmentation.py --n-clusters 5
```

**Run in batch mode (no interactive plots):**
```bash
python src/customer_segmentation.py --no-interactive
```

**Enable verbose logging:**
```bash
python src/customer_segmentation.py --verbose
```

**Adjust outlier detection sensitivity:**
```bash
python src/customer_segmentation.py --contamination 0.1
```

**Combined example:**
```bash
python src/customer_segmentation.py \
  --input data/CUSTOMERS_CLEAN.csv \
  --output-dir results/ \
  --n-clusters 4 \
  --no-interactive \
  --verbose
```

### Command Line Options

```
usage: customer_segmentation.py [-h] [--input INPUT] [--output-dir OUTPUT_DIR]
                                [--n-clusters N_CLUSTERS] [--no-interactive]
                                [--verbose] [--contamination CONTAMINATION]

options:
  -h, --help            Show help message and exit
  --input INPUT         Input CSV file path (default: data/CUSTOMERS_CLEAN.csv)
  --output-dir OUTPUT_DIR
                        Output directory for results (default: results)
  --n-clusters N_CLUSTERS
                        Number of clusters (auto-detected if not specified)
  --no-interactive      Run in batch mode without showing plots
  --verbose             Enable verbose logging
  --contamination CONTAMINATION
                        Expected proportion of outliers (default: 0.05)
```

### What the Script Does

The script will:
- Load and preprocess the data
- Perform exploratory data analysis
- Determine optimal number of clusters (using Elbow, Silhouette, and other metrics)
- Apply clustering algorithms (K-means and Hierarchical)
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

- pandas >= 1.5.0, < 3.0.0
- numpy >= 1.21.0, < 2.0.0
- matplotlib >= 3.5.0, < 4.0.0
- seaborn >= 0.11.0, < 1.0.0
- scikit-learn >= 1.1.0, < 2.0.0
- scipy >= 1.9.0, < 2.0.0
- pyyaml >= 6.0, < 7.0.0
- pytest >= 7.0.0, < 9.0.0 (for testing)
- pytest-cov >= 4.0.0, < 6.0.0 (for coverage)

## Testing

Run the unit tests:
```bash
python -m pytest tests/
```

Run tests with coverage:
```bash
python -m pytest tests/ --cov=src --cov-report=html
```

## Configuration

You can customize analysis parameters by editing `config.yaml`:

```yaml
data:
  input_path: "data/CUSTOMERS_CLEAN.csv"
  output_dir: "results"

clustering:
  min_clusters: 2
  max_clusters: 8
  optimal_k: null  # Set to override auto-detection

outlier_detection:
  contamination: 0.05

visualization:
  dpi: 300
  interactive: true
```

## Data Privacy & Security

⚠️ **IMPORTANT**: This tool processes customer personally identifiable information (PII):
- Customer IDs
- Birth dates
- Purchase history

**Before using real customer data:**
1. Ensure you have proper authorization
2. Anonymize/pseudonymize identifiers as needed
3. Comply with GDPR, CCPA, or applicable data privacy regulations
4. Never commit sensitive data to version control
5. Store data securely with appropriate access controls

For demonstration purposes, use synthetic or anonymized data.

## Troubleshooting

**Error: "Data file not found"**
- Ensure `data/CUSTOMERS_CLEAN.csv` exists
- Check file path with `--input` argument
- See `data/README.md` for data format requirements

**Error: "Missing required columns"**
- Verify CSV has all required columns (see Data Requirements section)
- Check column names match exactly (case-sensitive)

**Plots not showing**
- Run with `--no-interactive` for batch mode
- Check matplotlib backend configuration

**Memory errors**
- The script automatically samples large datasets for visualization
- Adjust sampling parameters in `config.yaml`

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and add tests
4. Ensure tests pass: `python -m pytest tests/`
5. Commit your changes (`git commit -m 'Add some amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

**Matthew Russell**

## Changelog

### Version 1.0.0 (2024)
- Initial release with K-means and hierarchical clustering
- Comprehensive CLI with argparse
- Unit tests with pytest
- Type hints throughout codebase
- Logging instead of print statements
- Configuration file support
- Multiple clustering validation metrics
- Adaptive sampling for large datasets
- Interactive and batch modes
- Improved error handling and validation
- Data privacy warnings and documentation

## Acknowledgments

- Dataset provided for educational purposes
- Scikit-learn community for excellent machine learning tools
- Matplotlib and Seaborn for visualization capabilities
