"""
Data initialization script for E-Commerce Intelligence Suite
Processes the Online Retail.xlsx file and loads it into the database
"""

import pandas as pd
import sys
import os

# Add utils to path
sys.path.append('utils')
sys.path.append('models')

from utils.database_setup import DatabaseManager
from utils.data_processing import DataProcessor

def initialize_database():
    """Initialize database with sample data from Excel file"""
    print("🚀 Starting E-Commerce Intelligence Suite Data Initialization")
    print("=" * 60)
    
    # Check if Excel file exists
    excel_file = "Online Retail.xlsx"
    if not os.path.exists(excel_file):
        print(f"❌ Excel file '{excel_file}' not found!")
        print("Please ensure the Online Retail.xlsx file is in the project directory.")
        return False
    
    try:
        # Initialize database manager
        print("📊 Initializing database...")
        db = DatabaseManager()
        
        # Initialize data processor
        processor = DataProcessor()
        
        # Load and process Excel data
        print("📖 Loading Excel file...")
        df = pd.read_excel(excel_file)
        print(f"   Loaded {len(df)} records from Excel")
        
        # Clean the data
        print("🧹 Cleaning data...")
        clean_df = processor.clean_retail_data(df)
        
        if len(clean_df) == 0:
            print("❌ No valid data remaining after cleaning!")
            return False
        
        # Load data into database using the cleaned DataFrame
        print("💾 Loading data into database...")
        
        # Load dimension tables
        print("   Loading products...")
        products = clean_df[['StockCode', 'Description', 'UnitPrice']].drop_duplicates()
        products.columns = ['ProductID', 'Description', 'UnitPrice']
        products.to_sql('DimProduct', db.conn, if_exists='replace', index=False)
        
        print("   Loading customers...")
        customers = clean_df[['CustomerID', 'Country']].drop_duplicates()
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
            'InvoiceNo', 'StockCode', 'CustomerID', 
            'InvoiceDate', 'Quantity', 'TotalPrice'
        ]].copy()
        
        fact_sales['DateID'] = fact_sales['InvoiceDate'].dt.strftime('%Y-%m-%d')
        fact_sales['ProductID'] = fact_sales['StockCode']
        
        fact_sales = fact_sales[[
            'InvoiceNo', 'ProductID', 'CustomerID', 
            'DateID', 'Quantity', 'TotalPrice'
        ]]
        
        fact_sales.to_sql('FactSales', db.conn, if_exists='replace', index=False)
        
        # Generate summary statistics
        print("\n📈 Data Loading Summary:")
        print(f"   Products: {len(products)}")
        print(f"   Customers: {len(customers)}")
        print(f"   Date range: {dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}")
        print(f"   Sales transactions: {len(fact_sales)}")
        print(f"   Total revenue: ${clean_df['TotalPrice'].sum():,.2f}")
        
        # Create sample CSV for users
        print("\n📄 Creating sample CSV file...")
        sample_data = clean_df.head(1000)  # First 1000 records as sample
        sample_data.to_csv('data/sample_retail_data.csv', index=False)
        print("   Sample CSV saved to data/sample_retail_data.csv")
        
        # Close database connection
        db.close()
        
        print("\n✅ Database initialization completed successfully!")
        print("🎯 Ready to run the Streamlit application!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during initialization: {str(e)}")
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
