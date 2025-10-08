"""
Data initialization script for E-Commerce Intelligence Suite
Processes any CSV file and loads it into the database with proper cleaning and validation
"""

import pandas as pd
import sys
import os
from datetime import datetime

# Add modules to path
sys.path.append('utils')
sys.path.append('models')

# Import configuration
from config import (
    CSV_FILE_PATH, COLUMN_MAPPING, REQUIRED_COLUMNS, OPTIONAL_COLUMNS,
    VALIDATION_RULES, DATABASE_PATH, SAMPLE_DATA_INFO, validate_config
)
from utils.database_setup import DatabaseManager
from utils.data_processing import DataProcessor

def print_header():
    """Print initialization header"""
    print("🚀 E-Commerce Intelligence Suite - Data Initialization")
    print("=" * 65)
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Data Source: {CSV_FILE_PATH}")
    print(f"🗄️ Database: {DATABASE_PATH}")
    print("=" * 65)

def validate_csv_file(file_path):
    """Validate that the CSV file exists and has required columns"""
    print("🔍 Validating CSV file...")
    
    if not os.path.exists(file_path):
        print(f"❌ CSV file not found: {file_path}")
        print("\n📝 Please update the CSV_FILE_PATH in config.py")
        print("   Example: CSV_FILE_PATH = 'path/to/your/sales_data.csv'")
        return False
    
    try:
        # Use utility function for encoding detection
        processor = DataProcessor()
        df_sample, detected_encoding = processor.read_csv_with_encoding_detection(file_path, nrows=5)
        print(f"✅ CSV file found with {len(df_sample.columns)} columns (encoding: {detected_encoding})")
        print(f"   Columns: {list(df_sample.columns)}")
        
        # Check if required columns can be mapped
        missing_columns = []
        for required_col, csv_col in COLUMN_MAPPING.items():
            if csv_col not in df_sample.columns:
                missing_columns.append(f"{required_col} -> {csv_col}")
        
        if missing_columns:
            print("❌ Missing required columns in CSV:")
            for col in missing_columns:
                print(f"   - {col}")
            print("\n📝 Please update COLUMN_MAPPING in config.py to match your CSV columns")
            return False
        
        print("✅ All required columns found in CSV")
        return True
        
    except Exception as e:
        print(f"❌ Error reading CSV file: {str(e)}")
        return False

def load_and_map_csv(file_path):
    """Load CSV file and map columns to standard format"""
    print("📖 Loading and mapping CSV data...")
    
    try:
        # Use utility function for encoding detection
        processor = DataProcessor()
        df, detected_encoding = processor.read_csv_with_encoding_detection(file_path)
        print(f"   Loaded {len(df)} rows from CSV (encoding: {detected_encoding})")
        
        # Map columns to standard names
        reverse_mapping = {v: k for k, v in COLUMN_MAPPING.items()}
        df_mapped = df.rename(columns=reverse_mapping)
        
        # Keep only the columns we need
        available_columns = [col for col in COLUMN_MAPPING.keys() if col in df_mapped.columns]
        df_final = df_mapped[available_columns].copy()
        
        print(f"   Mapped to {len(df_final.columns)} standard columns")
        print(f"   Columns: {list(df_final.columns)}")
        
        return df_final
        
    except Exception as e:
        print(f"❌ Error loading CSV: {str(e)}")
        return None

def advanced_data_cleaning(df):
    """Advanced data cleaning with configurable rules"""
    print("🧹 Performing advanced data cleaning...")
    original_count = len(df)
    
    # Step 1: Remove rows with missing critical data
    before_count = len(df)
    df = df.dropna(subset=REQUIRED_COLUMNS)
    print(f"   Removed {before_count - len(df)} rows with missing required data")
    
    # Step 2: Remove cancelled orders if configured
    if VALIDATION_RULES.get('remove_cancelled_orders', True):
        before_count = len(df)
        df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
        print(f"   Removed {before_count - len(df)} cancelled orders")
    
    # Step 3: Validate quantity and price ranges
    before_count = len(df)
    df = df[
        (df['Quantity'] >= VALIDATION_RULES.get('min_quantity', 0)) &
        (df['UnitPrice'] >= VALIDATION_RULES.get('min_price', 0)) &
        (df['UnitPrice'] <= VALIDATION_RULES.get('max_price', 10000))
    ]
    print(f"   Removed {before_count - len(df)} rows with invalid quantity/price")
    
    # Step 4: Clean product descriptions
    if 'Description' in df.columns:
        before_count = len(df)
        df['Description'] = df['Description'].str.strip().str.upper()
        min_desc_length = VALIDATION_RULES.get('min_description_length', 3)
        df = df[df['Description'].str.len() >= min_desc_length]
        print(f"   Cleaned descriptions, removed {before_count - len(df)} with short descriptions")
    
    # Step 5: Parse and validate dates
    if 'InvoiceDate' in df.columns:
        before_count = len(df)
        try:
            df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
            df = df.dropna(subset=['InvoiceDate'])
            print(f"   Parsed dates, removed {before_count - len(df)} with invalid dates")
        except Exception as e:
            print(f"   ⚠️ Date parsing warning: {str(e)}")
    
    # Step 6: Handle optional columns
    for col in OPTIONAL_COLUMNS:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                print(f"   Column '{col}' has {null_count} missing values (keeping as optional)")
    
    # Step 7: Calculate total price
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    
    print(f"✅ Data cleaning completed: {len(df)} clean records from {original_count} original ({len(df)/original_count*100:.1f}% retained)")
    return df

def initialize_database():
    """Initialize database with CSV data"""
    print_header()
    
    # Step 1: Validate configuration
    print("🔧 Validating configuration...")
    config_errors = validate_config()
    if config_errors:
        print("❌ Configuration errors found:")
        for error in config_errors:
            print(f"   - {error}")
        return False
    print("✅ Configuration is valid")
    
    # Step 2: Validate CSV file
    if not validate_csv_file(CSV_FILE_PATH):
        print("\n📋 Expected CSV format:")
        print(SAMPLE_DATA_INFO)
        return False
    
    # Step 3: Load and map CSV data
    df = load_and_map_csv(CSV_FILE_PATH)
    if df is None:
        return False
    
    # Step 4: Advanced data cleaning
    clean_df = advanced_data_cleaning(df)
    if len(clean_df) == 0:
        print("❌ No valid data remaining after cleaning!")
        return False
    
    try:
        # Initialize database manager
        print("📊 Initializing database...")
        db = DatabaseManager(DATABASE_PATH)
        
        # Step 5: Load data into database
        print("💾 Loading data into database...")
        
        # Load dimension tables
        print("   Loading products...")
        products = clean_df[['StockCode', 'Description', 'UnitPrice']].drop_duplicates()
        products.columns = ['ProductID', 'Description', 'UnitPrice']
        products.to_sql('DimProduct', db.conn, if_exists='replace', index=False)
        
        print("   Loading customers...")
        if 'CustomerID' in clean_df.columns:
            customers = clean_df[['CustomerID'] + (['Country'] if 'Country' in clean_df.columns else [])].drop_duplicates()
            if 'Country' not in customers.columns:
                customers['Country'] = 'Unknown'
        else:
            # Create dummy customer data if CustomerID is missing
            customers = pd.DataFrame({
                'CustomerID': ['GUEST'],
                'Country': ['Unknown']
            })
        customers.to_sql('DimCustomer', db.conn, if_exists='replace', index=False)
        
        print("   Loading dates...")
        dates = clean_df['InvoiceDate'].drop_duplicates().reset_index(drop=True)
        date_dim = pd.DataFrame({
            'DateID': dates.dt.strftime('%Y-%m-%d'),
            'Date': dates.dt.strftime('%Y-%m-%d'),
            'Month': dates.dt.month,
            'Year': dates.dt.year,
            'Weekday': dates.dt.weekday
        })
        date_dim.to_sql('DimDate', db.conn, if_exists='replace', index=False)
        
        print("   Loading sales transactions...")
        fact_sales = clean_df[[
            'InvoiceNo', 'StockCode', 
            'CustomerID' if 'CustomerID' in clean_df.columns else 'InvoiceNo',
            'InvoiceDate', 'Quantity', 'TotalPrice'
        ]].copy()
        
        # Handle missing CustomerID
        if 'CustomerID' not in clean_df.columns:
            fact_sales['CustomerID'] = 'GUEST'
        
        fact_sales['DateID'] = fact_sales['InvoiceDate'].dt.strftime('%Y-%m-%d')
        fact_sales['ProductID'] = fact_sales['StockCode']
        
        fact_sales = fact_sales[[
            'InvoiceNo', 'ProductID', 'CustomerID', 
            'DateID', 'Quantity', 'TotalPrice'
        ]]
        
        fact_sales.to_sql('FactSales', db.conn, if_exists='replace', index=False)
        
        # Step 6: Generate summary statistics
        print("\n📈 Data Loading Summary:")
        print(f"   📦 Products: {len(products)}")
        print(f"   👥 Customers: {len(customers)}")
        print(f"   📅 Date range: {dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}")
        print(f"   💰 Sales transactions: {len(fact_sales)}")
        print(f"   💵 Total revenue: ${clean_df['TotalPrice'].sum():,.2f}")
        print(f"   📊 Average order value: ${clean_df.groupby('InvoiceNo')['TotalPrice'].sum().mean():.2f}")
        
        # Step 7: Create updated sample CSV
        print("\n📄 Creating updated sample CSV...")
        os.makedirs('data', exist_ok=True)
        sample_size = min(1000, len(clean_df))
        sample_data = clean_df.head(sample_size)
        sample_data.to_csv('data/sample_retail_data.csv', index=False)
        print(f"   Sample CSV saved: data/sample_retail_data.csv ({sample_size} records)")
        
        # Close database connection
        db.close()
        
        print("\n" + "=" * 65)
        print("✅ DATABASE INITIALIZATION COMPLETED SUCCESSFULLY!")
        print("=" * 65)
        print("🎯 Next steps:")
        print("   1. Run: streamlit run app.py")
        print("   2. Open your browser to the provided URL")
        print("   3. Start exploring your e-commerce data!")
        print("=" * 65)
        
        return True
        
    except Exception as e:
        print(f"❌ Error during database initialization: {str(e)}")
        print("\n🔧 Troubleshooting tips:")
        print("   1. Check that your CSV file path is correct in config.py")
        print("   2. Verify your CSV has the required columns")
        print("   3. Ensure you have write permissions in the data directory")
        return False

if __name__ == "__main__":
    success = initialize_database()
    if success:
        print("\n🚀 Next steps:")
        print("   1. Run: streamlit run app.py")
        print("   2. Open your browser to the provided URL")
        print("   3. Start exploring your e-commerce data!")
    else:
        print("\n❌ Initialization failed. Please check the error messages above.")
