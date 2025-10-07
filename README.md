# 🛒 E-Commerce Intelligence Suite

**The Smart Manager's Dashboard for Data-Driven Decisions**

A powerful Streamlit-based analytics dashboard that helps e-commerce managers make smarter inventory and sales decisions through:

- 🛒 **Cross-Selling Engine** - Discover which products are bought together using Market Basket Analysis
- 📈 **Inventory Forecaster** - Predict future demand using ARIMA time series forecasting
- 📤 **Upload Module** - Analyze your own sales data with easy CSV upload

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Your own CSV sales data (see format requirements below)

### 🎯 Setup Options

Choose your preferred setup method:

#### **Option 1: Automated Setup (Recommended) 🤖**

For beginners or quick setup:

```bash
git clone https://github.com/equaan/E-Commerce-Intelligence-Suite.git
cd E-Commerce-Intelligence-Suite
python setup.py
```

The interactive setup wizard will:
- ✅ Check your Python version
- ✅ Install all dependencies automatically
- ✅ Help you configure your CSV file path
- ✅ Initialize the database with your data
- ✅ Optionally start the application

#### **Option 2: Manual Setup (Advanced) ⚙️**

For users who prefer manual control:

1. **Clone the repository**
   ```bash
   git clone https://github.com/equaan/E-Commerce-Intelligence-Suite.git
   cd E-Commerce-Intelligence-Suite
   ```

2. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure your data source**
   - Open `config.py` in a text editor
   - Update `CSV_FILE_PATH` to point to your CSV file:
   ```python
   CSV_FILE_PATH = "path/to/your/sales_data.csv"
   ```
   - If your CSV has different column names, update `COLUMN_MAPPING` accordingly

4. **Initialize the database with your data**
   ```bash
   python initialize_data.py
   ```

5. **Run the Streamlit application**
   ```bash
   streamlit run app.py
   ```

6. **Open your browser** to the provided URL (usually http://localhost:8501)

## 📊 CSV Data Requirements

Your CSV file **must contain** these columns (exact names can be different, just update `config.py`):

### Required Columns:
| Column | Description | Example Values |
|--------|-------------|----------------|
| `InvoiceNo` | Transaction/Order ID | "536365", "536366" |
| `StockCode` | Product ID/SKU | "85123A", "71053" |
| `Description` | Product name | "WHITE HANGING HEART T-LIGHT HOLDER" |
| `Quantity` | Number of items sold | 6, 8, 2 |
| `InvoiceDate` | Transaction date | "2010-12-01 08:26:00" |
| `UnitPrice` | Price per item | 2.55, 3.39 |

### Optional Columns:
| Column | Description | Example Values |
|--------|-------------|----------------|
| `CustomerID` | Customer identifier | "17850", "13047" |
| `Country` | Customer country | "United Kingdom", "France" |

### Sample CSV Format:
```csv
InvoiceNo,StockCode,Description,Quantity,InvoiceDate,UnitPrice,CustomerID,Country
536365,85123A,WHITE HANGING HEART T-LIGHT HOLDER,6,2010-12-01 08:26:00,2.55,17850,United Kingdom
536365,71053,WHITE METAL LANTERN,6,2010-12-01 08:26:00,3.39,17850,United Kingdom
536366,22633,HAND WARMER UNION JACK,6,2010-12-01 08:28:00,1.85,17850,United Kingdom
```

## ⚙️ Configuration Guide

### Step 1: Update CSV Path
Edit `config.py` and change the `CSV_FILE_PATH`:

```python
# Examples of different data sources:
CSV_FILE_PATH = "data/my_sales_data.csv"                    # Local file
CSV_FILE_PATH = "C:/Users/YourName/Downloads/sales.csv"     # Full path
CSV_FILE_PATH = "https://example.com/sales_data.csv"       # URL (if accessible)
```

### Step 2: Map Your Column Names
If your CSV has different column names, update the `COLUMN_MAPPING`:

```python
COLUMN_MAPPING = {
    'InvoiceNo': 'order_id',        # Your CSV has 'order_id' instead of 'InvoiceNo'
    'StockCode': 'product_sku',     # Your CSV has 'product_sku' instead of 'StockCode'
    'Description': 'product_name',   # Your CSV has 'product_name' instead of 'Description'
    'Quantity': 'qty',              # Your CSV has 'qty' instead of 'Quantity'
    'InvoiceDate': 'purchase_date', # Your CSV has 'purchase_date' instead of 'InvoiceDate'
    'UnitPrice': 'price',           # Your CSV has 'price' instead of 'UnitPrice'
    'CustomerID': 'customer_id',    # Your CSV has 'customer_id' instead of 'CustomerID'
}
```

### Step 3: Adjust Data Validation (Optional)
Modify validation rules in `config.py` if needed:

```python
VALIDATION_RULES = {
    'min_quantity': 1,              # Change to 1 if you don't want to include returns
    'min_price': 0.01,              # Minimum price threshold
    'max_price': 5000,              # Maximum price threshold (adjust for your products)
    'remove_cancelled_orders': True, # Set to False if you want to keep cancelled orders
}
```


## 📁 Sample Datasets

Don't have your own data yet? Try these popular e-commerce datasets:

### 🛍️ **Online Retail Dataset (Recommended)**
- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Online+Retail)
- **Description**: UK-based online retail transactions (2010-2011)
- **Size**: ~540K transactions
- **Perfect for**: Testing all features of the suite

### 🛒 **E-commerce Data**
- **Source**: [Kaggle E-commerce Datasets](https://www.kaggle.com/datasets?search=ecommerce+sales)
- **Various datasets** available with different formats
- **Note**: May require column mapping in `config.py`

### 🏪 **Retail Sales Data**
- **Source**: [Kaggle Retail Analytics](https://www.kaggle.com/datasets?search=retail+sales)
- **Multiple options** for different retail scenarios
- **Good for**: Testing forecasting capabilities

### 📊 **Custom Data Requirements**
Your data should represent **transactional sales records** where each row is a product sold in a transaction. The system works best with:
- **Multiple products per transaction** (for cross-selling analysis)
- **Time series data** spanning several months (for forecasting)
- **Consistent product identifiers** (for accurate analysis)

## 📊 Features

### 🏠 Home Dashboard
- Business overview with key metrics
- Daily sales trends
- Top-performing products
- Key insights and analytics

### 🛒 Cross-Selling Engine
- Market Basket Analysis using Apriori algorithm
- Product recommendation system
- Association rules with confidence, lift, and support metrics
- Interactive parameter tuning
- Visual charts and insights

### 📈 Inventory Forecaster
- ARIMA-based demand forecasting
- Configurable forecast periods (7-90 days)
- Confidence intervals and trend analysis
- Stock level recommendations
- Model accuracy metrics (MAPE)

### 📤 Upload Your Data
- CSV file validation and processing
- Sample data format download
- Real-time data cleaning and analysis
- Support for custom datasets

## 🗂️ Project Structure

```
DWM Project/
├── app.py                      # Main Streamlit application
├── initialize_data.py          # Database initialization script
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── progress.md                 # Development progress tracker
├── PRD.md                      # Product Requirements Document
├── Online Retail.xlsx          # Sample dataset
├── data/                       # Data directory
│   ├── ecommerce_warehouse.db  # SQLite database
│   └── sample_retail_data.csv  # Sample CSV export
├── models/                     # ML models
│   ├── market_basket.py        # Market Basket Analysis
│   └── inventory_forecaster.py # ARIMA Forecasting
└── utils/                      # Utility modules
    ├── database_setup.py       # Database management
    └── data_processing.py      # Data cleaning & validation
```

## 🎯 Usage Guide

### Getting Started
1. **Home Page**: View your business overview and key metrics
2. **Cross-Selling**: Select a product to get recommendations for cross-selling
3. **Forecasting**: Choose a product and forecast period to predict demand
4. **Upload Data**: Use your own CSV data for personalized analysis

### Data Format
Your CSV file should contain these columns:
- `InvoiceNo`: Transaction identifier
- `StockCode`: Product identifier
- `Description`: Product name/description
- `Quantity`: Number of items sold
- `InvoiceDate`: Transaction date (YYYY-MM-DD format)
- `UnitPrice`: Price per unit
- `CustomerID`: Customer identifier

### Tips for Best Results
- **Cross-Selling**: Lower the minimum support for more recommendations
- **Forecasting**: Use products with at least 30 days of sales history
- **Data Quality**: Clean data produces better insights

## 🔧 Technical Details

### Tech Stack
- **Frontend**: Streamlit, Plotly
- **Backend**: Python, Pandas, NumPy
- **ML Libraries**: mlxtend (Apriori), statsmodels (ARIMA)
- **Database**: SQLite with star schema
- **Deployment**: Streamlit Cloud ready

### Database Schema
- **FactSales**: Main transaction table
- **DimProduct**: Product dimension
- **DimCustomer**: Customer dimension  
- **DimDate**: Date dimension

### Algorithms Used
- **Apriori Algorithm**: For market basket analysis and association rules
- **ARIMA Model**: For time series forecasting with automatic parameter selection

## 📈 Performance Notes

- Optimized for datasets up to 100K transactions
- Uses caching for improved performance
- Responsive design for various screen sizes
- Real-time analysis with progress indicators

## 🚀 Deployment

### Local Development
```bash
streamlit run app.py
```

### Streamlit Cloud Deployment
1. Push code to GitHub repository
2. Connect to Streamlit Cloud
3. Deploy with one click
4. Share your dashboard URL

## 🛠️ Troubleshooting

### Common Issues

**"No data available" error**
- Run `python initialize_data.py` first
- Ensure `Online Retail.xlsx` is in the project directory

**Import errors**
- Install requirements: `pip install -r requirements.txt`
- Check Python version (3.8+ required)

**Slow performance**
- Reduce dataset size for testing
- Adjust analysis parameters (support, confidence thresholds)

**Forecast errors**
- Ensure product has sufficient historical data (30+ days)
- Check for data quality issues

## 📝 Version History

- **v1.0** - Initial release with core features
  - Cross-selling engine with Apriori algorithm
  - ARIMA-based inventory forecasting
  - Data upload and validation
  - Interactive Streamlit dashboard

## 🤝 Contributing

This project is part of an academic assignment. For suggestions or improvements:

1. Document issues in the progress.md file
2. Test thoroughly before making changes
3. Follow the existing code structure and style

## 📄 License

This project is for educational purposes as part of a Data Warehouse and Mining course.

## 🙏 Acknowledgments

- **Dataset**: Online Retail Dataset (UCI Machine Learning Repository)
- **Libraries**: Streamlit, Plotly, mlxtend, statsmodels, pandas
- **Inspiration**: Real-world e-commerce analytics needs

---

**Built with ❤️ for smarter e-commerce decisions**

For questions or support, refer to the PRD.md file for detailed specifications and requirements.