"""
Database setup and management utilities for E-Commerce Intelligence Suite
Implements star schema with fact and dimension tables
"""

import sqlite3
import pandas as pd
from datetime import datetime
import os

class DatabaseManager:
    def __init__(self, db_path="data/ecommerce_warehouse.db"):
        """Initialize database manager with SQLite connection"""
        self.db_path = db_path
        self.ensure_data_directory()
        self.conn = sqlite3.connect(db_path)
        self.create_tables()
    
    def ensure_data_directory(self):
        """Create data directory if it doesn't exist"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
    
    def create_tables(self):
        """Create star schema tables if they don't exist"""
        cursor = self.conn.cursor()
        
        # Dimension Tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS DimProduct (
                ProductID TEXT PRIMARY KEY,
                Description TEXT,
                UnitPrice REAL
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS DimCustomer (
                CustomerID TEXT PRIMARY KEY,
                Country TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS DimDate (
                DateID TEXT PRIMARY KEY,
                Date TEXT,
                Month INTEGER,
                Year INTEGER,
                Weekday INTEGER
            )
        """)
        
        # Fact Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS FactSales (
                InvoiceNo TEXT,
                ProductID TEXT,
                CustomerID TEXT,
                DateID TEXT,
                Quantity INTEGER,
                TotalPrice REAL,
                FOREIGN KEY (ProductID) REFERENCES DimProduct(ProductID),
                FOREIGN KEY (CustomerID) REFERENCES DimCustomer(CustomerID),
                FOREIGN KEY (DateID) REFERENCES DimDate(DateID)
            )
        """)
        
        self.conn.commit()
        print("✅ Database tables created successfully!")
    
    def load_data_from_excel(self, excel_path):
        """Load and process data from Excel file into star schema"""
        try:
            # Read Excel file
            df = pd.read_excel(excel_path)
            print(f"📊 Loaded {len(df)} records from Excel file")
            
            # Clean data
            df = self.clean_data(df)
            print(f"🧹 After cleaning: {len(df)} records")
            
            # Load into dimension tables
            self.load_dim_product(df)
            self.load_dim_customer(df)
            self.load_dim_date(df)
            
            # Load into fact table
            self.load_fact_sales(df)
            
            print("✅ Data loaded successfully into warehouse!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            return False
    
    def clean_data(self, df):
        """Clean and validate the raw data"""
        # Remove cancelled orders (negative quantities)
        df = df[df['Quantity'] > 0]
        
        # Remove missing customer IDs
        df = df.dropna(subset=['CustomerID'])
        
        # Remove missing descriptions
        df = df.dropna(subset=['Description'])
        
        # Calculate total price
        df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
        
        # Convert InvoiceDate to datetime
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        
        return df
    
    def load_dim_product(self, df):
        """Load unique products into DimProduct table"""
        products = df[['StockCode', 'Description', 'UnitPrice']].drop_duplicates()
        products.columns = ['ProductID', 'Description', 'UnitPrice']
        
        products.to_sql('DimProduct', self.conn, if_exists='replace', index=False)
        print(f"📦 Loaded {len(products)} products")
    
    def load_dim_customer(self, df):
        """Load unique customers into DimCustomer table"""
        customers = df[['CustomerID', 'Country']].drop_duplicates()
        
        customers.to_sql('DimCustomer', self.conn, if_exists='replace', index=False)
        print(f"👥 Loaded {len(customers)} customers")
    
    def load_dim_date(self, df):
        """Load unique dates into DimDate table"""
        dates = df['InvoiceDate'].drop_duplicates().reset_index(drop=True)
        
        date_dim = pd.DataFrame({
            'DateID': dates.dt.strftime('%Y-%m-%d'),
            'Date': dates.dt.strftime('%Y-%m-%d'),
            'Month': dates.dt.month,
            'Year': dates.dt.year,
            'Weekday': dates.dt.weekday
        })
        
        date_dim.to_sql('DimDate', self.conn, if_exists='replace', index=False)
        print(f"📅 Loaded {len(date_dim)} dates")
    
    def load_fact_sales(self, df):
        """Load sales transactions into FactSales table"""
        fact_sales = df[[
            'InvoiceNo', 'StockCode', 'CustomerID', 
            'InvoiceDate', 'Quantity', 'TotalPrice'
        ]].copy()
        
        fact_sales['DateID'] = fact_sales['InvoiceDate'].dt.strftime('%Y-%m-%d')
        fact_sales['ProductID'] = fact_sales['StockCode']
        
        fact_sales = fact_sales[[
            'InvoiceNo', 'ProductID', 'CustomerID', 
            'DateID', 'Quantity', 'TotalPrice'
        ]]
        
        fact_sales.to_sql('FactSales', self.conn, if_exists='replace', index=False)
        print(f"💰 Loaded {len(fact_sales)} sales transactions")
    
    def get_sales_data(self):
        """Get sales data for analysis"""
        query = """
        SELECT 
            fs.InvoiceNo,
            fs.ProductID,
            dp.Description,
            fs.CustomerID,
            fs.DateID,
            fs.Quantity,
            fs.TotalPrice,
            dp.UnitPrice
        FROM FactSales fs
        JOIN DimProduct dp ON fs.ProductID = dp.ProductID
        """
        return pd.read_sql_query(query, self.conn)
    
    def get_product_list(self):
        """Get list of all products"""
        query = "SELECT ProductID, Description FROM DimProduct ORDER BY Description"
        return pd.read_sql_query(query, self.conn)
    
    def close(self):
        """Close database connection"""
        self.conn.close()

if __name__ == "__main__":
    # Test database setup
    db = DatabaseManager()
    print("Database setup completed!")
