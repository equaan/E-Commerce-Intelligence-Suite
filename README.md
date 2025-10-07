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

### 🎯 Setup 

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



## 📊 Features (See USER_MANUAL.md for a complete feature overview)

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
├── setup.py                    # Installs deps, cleans data, inits DB, starts app
├── config.py                   # column mapping & validation
├── USER_MANUAL.md              # Feature usage guide
├── models/                     # ML models
│   ├── market_basket.py        # Market Basket Analysis
│   └── inventory_forecaster.py # ARIMA Forecasting
└── utils/                      # Utility modules
    ├── database_setup.py       # Database management
    └── data_processing.py      # Data cleaning & validation
```


## 🔧 Technical Details

### Tech Stack
- **Frontend**: Streamlit, Plotly
- **Backend**: Python, Pandas, NumPy
- **ML Libraries**: mlxtend (Apriori), statsmodels (ARIMA)
- **Database**: SQLite with star schema

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



## ⚙️ Configuration Guide (Only use when you have different column names and different type of dataset)

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


## 👤 Author
- [Mohammad Equaan Kacchi](https://www.linkedin.com/in/mohammad-equaan-kacchi-4a8a49290/)

## 🤝 Contributors
- [Hussain Anajwala](https://www.linkedin.com/in/hussain-anajwala-99435434a/)
