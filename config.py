"""
Configuration file for E-Commerce Intelligence Suite
Users can modify this file to specify their CSV data source and column mappings
"""

import os

# =============================================================================
# 📁 DATA SOURCE CONFIGURATION
# =============================================================================

# Path to your CSV file (modify this to point to your data)
# Examples:
# CSV_FILE_PATH = "data/your_sales_data.csv"
# CSV_FILE_PATH = "C:/Users/YourName/Downloads/online_retail.csv"
# CSV_FILE_PATH = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"

CSV_FILE_PATH = "data/sample_retail_data.csv"  # Default sample data

# =============================================================================
# 📊 COLUMN MAPPING CONFIGURATION
# =============================================================================

# Map your CSV columns to the required format
# Modify the RIGHT side to match your CSV column names
COLUMN_MAPPING = {
    # Required columns (must be present in your CSV)
    'InvoiceNo': 'InvoiceNo',           # Transaction/Order ID
    'StockCode': 'StockCode',           # Product ID/SKU
    'Description': 'Description',        # Product name/description
    'Quantity': 'Quantity',             # Number of items sold
    'InvoiceDate': 'InvoiceDate',       # Transaction date
    'UnitPrice': 'UnitPrice',           # Price per item
    'CustomerID': 'CustomerID',         # Customer identifier
    
    # Optional columns (will be used if present)
    'Country': 'Country',               # Customer country (optional)
}

# =============================================================================
# 🧹 DATA CLEANING CONFIGURATION
# =============================================================================

# Columns that must have values (no nulls allowed)
REQUIRED_COLUMNS = ['InvoiceNo', 'StockCode', 'Description', 'Quantity', 'UnitPrice']

# Columns that can have some missing values
OPTIONAL_COLUMNS = ['CustomerID', 'Country']

# Data validation rules
VALIDATION_RULES = {
    'min_quantity': 0,          # Minimum quantity (negative = returns/cancellations)
    'min_price': 0,             # Minimum unit price
    'max_price': 10000,         # Maximum unit price (adjust based on your products)
    'date_format': '%Y-%m-%d %H:%M:%S',  # Expected date format
    'remove_cancelled_orders': True,     # Remove orders starting with 'C'
    'min_description_length': 3,         # Minimum product description length
}

# =============================================================================
# 🗄️ DATABASE CONFIGURATION
# =============================================================================

# Database file location
DATABASE_PATH = "data/ecommerce_warehouse.db"

# =============================================================================
# 📈 ANALYSIS CONFIGURATION
# =============================================================================

# Default analysis parameters
DEFAULT_ANALYSIS_PARAMS = {
    'min_support': 0.01,        # Market basket analysis minimum support
    'min_confidence': 0.1,      # Market basket analysis minimum confidence
    'min_lift': 1.0,           # Market basket analysis minimum lift
    'forecast_days': 30,        # Default forecast period
    'min_data_points': 30,      # Minimum data points needed for forecasting
}

# =============================================================================
# 🎨 UI CONFIGURATION
# =============================================================================

# Streamlit app configuration
APP_CONFIG = {
    'page_title': 'E-Commerce Intelligence Suite',
    'page_icon': '🛒',
    'layout': 'wide',
    'sidebar_state': 'expanded'
}

# =============================================================================
# 📝 SAMPLE DATA INFORMATION
# =============================================================================

SAMPLE_DATA_INFO = """
Your CSV file should contain the following columns:

REQUIRED COLUMNS:
- InvoiceNo: Transaction/Order ID (e.g., "536365", "536366")
- StockCode: Product ID/SKU (e.g., "85123A", "71053")
- Description: Product name (e.g., "WHITE HANGING HEART T-LIGHT HOLDER")
- Quantity: Number of items sold (e.g., 6, 8, 2)
- InvoiceDate: Transaction date (e.g., "2010-12-01 08:26:00")
- UnitPrice: Price per item (e.g., 2.55, 3.39)

OPTIONAL COLUMNS:
- CustomerID: Customer identifier (e.g., "17850", "13047")
- Country: Customer country (e.g., "United Kingdom", "France")

EXAMPLE CSV FORMAT:
InvoiceNo,StockCode,Description,Quantity,InvoiceDate,UnitPrice,CustomerID,Country
536365,85123A,WHITE HANGING HEART T-LIGHT HOLDER,6,2010-12-01 08:26:00,2.55,17850,United Kingdom
536365,71053,WHITE METAL LANTERN,6,2010-12-01 08:26:00,3.39,17850,United Kingdom

ENCODING SUPPORT:
The system automatically detects and supports multiple encodings:
- UTF-8 (recommended)
- Latin-1 (ISO-8859-1)
- CP1252 (Windows-1252)
- UTF-16

If you get encoding errors:
1. Save your CSV as UTF-8 in Excel: File > Save As > CSV UTF-8
2. Or use a text editor like Notepad++ to convert encoding
"""

def get_csv_path():
    """Get the CSV file path, checking if it exists"""
    if os.path.exists(CSV_FILE_PATH):
        return CSV_FILE_PATH
    else:
        raise FileNotFoundError(f"CSV file not found at: {CSV_FILE_PATH}")

def validate_config():
    """Validate the configuration settings"""
    errors = []
    
    # Check if CSV file exists
    if not os.path.exists(CSV_FILE_PATH):
        errors.append(f"CSV file not found: {CSV_FILE_PATH}")
    
    # Check required column mappings
    required_keys = ['InvoiceNo', 'StockCode', 'Description', 'Quantity', 'InvoiceDate', 'UnitPrice']
    for key in required_keys:
        if key not in COLUMN_MAPPING:
            errors.append(f"Missing required column mapping: {key}")
    
    return errors

if __name__ == "__main__":
    # Test configuration
    print("🔧 Testing Configuration...")
    errors = validate_config()
    
    if errors:
        print("❌ Configuration Errors:")
        for error in errors:
            print(f"   - {error}")
    else:
        print("✅ Configuration is valid!")
        print(f"📁 CSV Path: {CSV_FILE_PATH}")
        print(f"🗄️ Database Path: {DATABASE_PATH}")
