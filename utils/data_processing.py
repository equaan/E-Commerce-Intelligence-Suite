"""
Data processing and ETL utilities for E-Commerce Intelligence Suite
Handles data cleaning, validation, and transformation
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sqlite3
import io

class DataProcessor:
    def __init__(self):
        """Initialize data processor"""
        self.required_columns = [
            'InvoiceNo', 'StockCode', 'Description', 
            'Quantity', 'InvoiceDate', 'UnitPrice', 'CustomerID'
        ]
    
    def read_csv_with_encoding_detection(self, file_path_or_buffer, **kwargs):
        """
        Read CSV file with automatic encoding detection
        Supports both file paths and file-like objects (for Streamlit uploads)
        """
        encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']
        
        for encoding in encodings_to_try:
            try:
                # Reset file pointer if it's a file-like object
                if hasattr(file_path_or_buffer, 'seek'):
                    file_path_or_buffer.seek(0)
                
                df = pd.read_csv(file_path_or_buffer, encoding=encoding, **kwargs)
                return df, encoding
                
            except UnicodeDecodeError:
                continue
            except Exception as e:
                if encoding == encodings_to_try[-1]:  # Last encoding to try
                    raise e
                continue
        
        raise ValueError("Could not read file with any supported encoding")
    
    def validate_csv_schema(self, df):
        """Validate that uploaded CSV has required columns"""
        missing_columns = []
        for col in self.required_columns:
            if col not in df.columns:
                missing_columns.append(col)
        
        if missing_columns:
            return False, f"Missing columns: {', '.join(missing_columns)}"
        
        return True, "Schema validation passed"
    
    def clean_retail_data(self, df):
        """Clean and validate retail transaction data"""
        print("🧹 Starting data cleaning process...")
        original_count = len(df)
        
        # Step 1: Remove rows with missing critical data
        df = df.dropna(subset=['InvoiceNo', 'StockCode', 'Description', 'CustomerID'])
        print(f"   Removed {original_count - len(df)} rows with missing critical data")
        
        # Step 2: Remove cancelled orders (InvoiceNo starting with 'C')
        df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
        print(f"   Removed cancelled orders, {len(df)} rows remaining")
        
        # Step 3: Remove negative quantities and zero prices
        df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
        print(f"   Removed invalid quantities/prices, {len(df)} rows remaining")
        
        # Step 4: Clean product descriptions
        df['Description'] = df['Description'].str.strip().str.upper()
        
        # Step 5: Ensure proper data types
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
        df['CustomerID'] = df['CustomerID'].astype(str)
        df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
        df['UnitPrice'] = pd.to_numeric(df['UnitPrice'], errors='coerce')
        
        # Step 6: Remove rows with invalid dates or prices after conversion
        df = df.dropna(subset=['InvoiceDate', 'Quantity', 'UnitPrice'])
        
        # Step 7: Calculate total price
        df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
        
        print(f"✅ Data cleaning completed: {len(df)} clean records from {original_count} original")
        return df
    
    def prepare_basket_data(self, df):
        """Prepare data for market basket analysis"""
        # Group by invoice to create baskets
        baskets = df.groupby('InvoiceNo')['Description'].apply(list).reset_index()
        baskets.columns = ['InvoiceNo', 'Items']
        
        # Filter baskets with at least 2 items
        baskets = baskets[baskets['Items'].apply(len) >= 2]
        
        print(f"📊 Prepared {len(baskets)} baskets for market basket analysis")
        return baskets
    
    def prepare_time_series_data(self, df):
        """Prepare data for time series forecasting"""
        # Check if we have StockCode or ProductID column
        product_col = 'StockCode' if 'StockCode' in df.columns else 'ProductID'
        date_col = 'InvoiceDate' if 'InvoiceDate' in df.columns else 'DateID'
        
        # Convert DateID to datetime if needed
        if date_col == 'DateID':
            df[date_col] = pd.to_datetime(df[date_col])
        
        # Convert date column to date only (remove time) to avoid duplicates
        df_copy = df.copy()
        df_copy[date_col] = pd.to_datetime(df_copy[date_col]).dt.date
        
        # Aggregate daily sales by product (sum all transactions per day)
        daily_sales = df_copy.groupby([product_col, 'Description', date_col]).agg({
            'Quantity': 'sum',
            'TotalPrice': 'sum'
        }).reset_index()
        
        # Convert date back to datetime for time series operations
        daily_sales[date_col] = pd.to_datetime(daily_sales[date_col])
        
        # Create date range for each product
        products = daily_sales[[product_col, 'Description']].drop_duplicates()
        date_range = pd.date_range(
            start=daily_sales[date_col].min(),
            end=daily_sales[date_col].max(),
            freq='D'
        )
        
        # Create complete time series with zero-filled missing dates
        complete_series = []
        for _, product in products.iterrows():
            product_data = daily_sales[
                daily_sales[product_col] == product[product_col]
            ].copy()
            
            # Set date as index and ensure no duplicates
            product_data = product_data.set_index(date_col)
            product_data = product_data[~product_data.index.duplicated(keep='first')]
            
            # Keep only the numeric columns we need for reindexing
            product_data = product_data[['Quantity', 'TotalPrice']]
            
            # Reindex to fill missing dates with 0
            product_series = product_data.reindex(date_range, fill_value=0)
            
            # Add product info after reindexing
            product_series['StockCode'] = product[product_col]  # Standardize to StockCode
            product_series['Description'] = product['Description']
            
            # Reset index to get Date as a column
            product_series = product_series.reset_index()
            
            # Now we should have exactly 5 columns: index(Date), Quantity, TotalPrice, StockCode, Description
            product_series.columns = ['Date', 'Quantity', 'TotalPrice', 'StockCode', 'Description']
            
            complete_series.append(product_series)
        
        result = pd.concat(complete_series, ignore_index=True)
        print(f"📈 Prepared time series data for {len(products)} products")
        return result
    
    def get_top_products(self, df, n=50):
        """Get top N products by total sales volume"""
        # Check if we have StockCode or ProductID column
        product_col = 'StockCode' if 'StockCode' in df.columns else 'ProductID'
        
        top_products = df.groupby([product_col, 'Description']).agg({
            'Quantity': 'sum',
            'TotalPrice': 'sum'
        }).reset_index()
        
        # Rename column to standardize
        if product_col == 'ProductID':
            top_products = top_products.rename(columns={'ProductID': 'StockCode'})
        
        top_products = top_products.sort_values('TotalPrice', ascending=False).head(n)
        return top_products
    
    def validate_upload_data(self, df):
        """Comprehensive validation for uploaded data"""
        issues = []
        
        # Check required columns
        valid_schema, schema_msg = self.validate_csv_schema(df)
        if not valid_schema:
            issues.append(schema_msg)
            return False, issues
        
        # Check data quality
        if len(df) == 0:
            issues.append("File is empty")
        
        if df['InvoiceNo'].isnull().sum() > len(df) * 0.1:
            issues.append("Too many missing invoice numbers (>10%)")
        
        if df['CustomerID'].isnull().sum() > len(df) * 0.2:
            issues.append("Too many missing customer IDs (>20%)")
        
        # Check date format
        try:
            pd.to_datetime(df['InvoiceDate'].dropna().head(100))
        except:
            issues.append("Invalid date format in InvoiceDate column")
        
        # Check numeric columns
        numeric_cols = ['Quantity', 'UnitPrice']
        for col in numeric_cols:
            if not pd.to_numeric(df[col], errors='coerce').notna().any():
                issues.append(f"Column {col} contains no valid numeric values")
        
        if issues:
            return False, issues
        else:
            return True, ["Data validation passed successfully"]

def create_sample_csv():
    """Create a sample CSV file for users to download"""
    sample_data = {
        'InvoiceNo': ['536365', '536365', '536366', '536366', '536367'],
        'StockCode': ['85123A', '71053', '84406B', '84029G', '85123A'],
        'Description': ['WHITE HANGING HEART T-LIGHT HOLDER', 'WHITE METAL LANTERN', 
                       'CREAM CUPID HEARTS COAT HANGER', 'KNITTED UNION FLAG HOT WATER BOTTLE',
                       'WHITE HANGING HEART T-LIGHT HOLDER'],
        'Quantity': [6, 6, 8, 6, 6],
        'InvoiceDate': ['2010-12-01 08:26:00', '2010-12-01 08:26:00', 
                       '2010-12-01 08:28:00', '2010-12-01 08:28:00', '2010-12-01 08:34:00'],
        'UnitPrice': [2.55, 3.39, 2.75, 3.39, 2.55],
        'CustomerID': ['17850', '17850', '13047', '13047', '13047']
    }
    
    sample_df = pd.DataFrame(sample_data)
    return sample_df

if __name__ == "__main__":
    # Test data processor
    processor = DataProcessor()
    sample = create_sample_csv()
    print("Sample data created:")
    print(sample.head())
