"""
E-Commerce Intelligence Suite - Main Streamlit Application
A data-driven dashboard for smart inventory and sales decisions
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
import psutil
from datetime import datetime, timedelta

# Add modules to path
sys.path.append('utils')
sys.path.append('models')

from utils.database_setup import DatabaseManager
from utils.data_processing import DataProcessor
from utils.cache_manager import clear_cache, get_cache_stats, cleanup_memory
from models.market_basket import MarketBasketAnalyzer
from models.inventory_forecaster import InventoryForecaster

# Page configuration
st.set_page_config(
    page_title="E-Commerce Intelligence Suite",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f4e79;
        margin: 0.5rem 0;
    }
    .insight-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #17a2b8;
        margin: 1rem 0;
    }
    .sidebar .sidebar-content {
        background-color: #1f4e79;
    }
</style>
""", unsafe_allow_html=True)

def load_data():
    """Load data from database with lightweight caching"""
    # Use session state for lightweight caching instead of heavy st.cache_data
    if 'cached_data' not in st.session_state or 'cached_products' not in st.session_state:
        try:
            db = DatabaseManager()
            data = db.get_sales_data()
            products = db.get_product_list()
            db.close()
            
            # Store in session state for this session only
            st.session_state.cached_data = data
            st.session_state.cached_products = products
            
            return data, products
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None, None
    else:
        return st.session_state.cached_data, st.session_state.cached_products

# Removed heavy caching functions - now using disk-based caching in models

def main():
    """Main application function"""
    
    # Initialize session state for performance tracking
    if 'last_analysis_params' not in st.session_state:
        st.session_state.last_analysis_params = {}
    
    # Header
    st.markdown('<h1 class="main-header">🛒 E-Commerce Intelligence Suite</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666; font-size: 1.2rem;">The Smart Manager\'s Dashboard for Data-Driven Decisions</p>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["🏠 Home", "🛒 Cross-Selling Engine", "📈 Inventory Forecaster", "📤 Upload Data"]
    )
    
    # Cache management section
    st.sidebar.markdown("---")
    st.sidebar.subheader("🗄️ Cache Management")
    
    cache_stats = get_cache_stats()
    if cache_stats.get('error'):
        st.sidebar.error("Cache stats unavailable")
    else:
        st.sidebar.metric("MBA Cache Files", cache_stats.get('mba_cache_files', 0))
        st.sidebar.metric("ARIMA Cache Files", cache_stats.get('arima_cache_files', 0))
        st.sidebar.metric("Cache Size (MB)", cache_stats.get('total_cache_size_mb', 0))
        
        if st.sidebar.button("🗑️ Clear All Cache"):
            if clear_cache():
                st.sidebar.success("Cache cleared!")
                cleanup_memory()
            else:
                st.sidebar.error("Failed to clear cache")
    
    # Memory usage indicator
    st.sidebar.markdown("---")
    st.sidebar.subheader("💾 Memory Usage")
    try:
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        memory_percent = process.memory_percent()
        
        st.sidebar.metric("RAM Usage (MB)", f"{memory_mb:.1f}")
        st.sidebar.metric("RAM Usage (%)", f"{memory_percent:.1f}%")
        
        if memory_percent > 80:
            st.sidebar.error("⚠️ High memory usage!")
        elif memory_percent > 60:
            st.sidebar.warning("⚠️ Moderate memory usage")
    except:
        st.sidebar.info("Memory stats unavailable")
    
    # Clean up memory when switching pages
    if 'current_page' not in st.session_state:
        st.session_state.current_page = page
    elif st.session_state.current_page != page:
        st.session_state.current_page = page
        # Clear heavy data from session state when switching pages
        if 'cached_data' in st.session_state:
            del st.session_state.cached_data
        if 'cached_products' in st.session_state:
            del st.session_state.cached_products
        cleanup_memory()
    
    # Load data
    data, products = load_data()
    
    if data is None or len(data) == 0:
        st.error("❌ No data available. Please run the data initialization script first.")
        st.code("python initialize_data.py")
        return
    
    # Page routing
    if page == "🏠 Home":
        show_home_page(data)
    elif page == "🛒 Cross-Selling Engine":
        show_cross_selling_page(data, products)
    elif page == "📈 Inventory Forecaster":
        show_forecasting_page(data, products)
    elif page == "📤 Upload Data":
        show_upload_page()

def show_home_page(data):
    """Display home page with overview statistics"""
    st.header("📊 Business Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_revenue = data['TotalPrice'].sum()
        st.metric("💰 Total Revenue", f"${total_revenue:,.2f}")
    
    with col2:
        total_orders = data['InvoiceNo'].nunique()
        st.metric("📋 Total Orders", f"{total_orders:,}")
    
    with col3:
        total_products = data['ProductID'].nunique()
        st.metric("📦 Unique Products", f"{total_products:,}")
    
    with col4:
        total_customers = data['CustomerID'].nunique()
        st.metric("👥 Customers", f"{total_customers:,}")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Daily Sales Trend")
        daily_sales = data.groupby('DateID')['TotalPrice'].sum().reset_index()
        daily_sales['DateID'] = pd.to_datetime(daily_sales['DateID'])
        
        fig = px.line(daily_sales, x='DateID', y='TotalPrice', 
                     title="Daily Revenue Trend")
        fig.update_layout(xaxis_title="Date", yaxis_title="Revenue ($)")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🏆 Top 10 Products by Revenue")
        top_products = data.groupby('Description')['TotalPrice'].sum().sort_values(ascending=False).head(10)
        
        fig = px.bar(x=top_products.values, y=top_products.index, orientation='h',
                    title="Top Products by Revenue")
        fig.update_layout(xaxis_title="Revenue ($)", yaxis_title="Product")
        st.plotly_chart(fig, use_container_width=True)
    
    # Insights
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.subheader("💡 Key Insights")
    
    avg_order_value = data.groupby('InvoiceNo')['TotalPrice'].sum().mean()
    best_month = data.groupby(pd.to_datetime(data['DateID']).dt.month)['TotalPrice'].sum().idxmax()
    month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                   7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    
    st.write(f"• **Average Order Value**: ${avg_order_value:.2f}")
    st.write(f"• **Best Performing Month**: {month_names.get(best_month, best_month)}")
    st.write(f"• **Data Period**: {data['DateID'].min()} to {data['DateID'].max()}")
    st.markdown('</div>', unsafe_allow_html=True)

def show_cross_selling_page(data, products):
    """Display cross-selling analysis page"""
    st.header("🛒 Cross-Selling Engine")
    st.write("Discover which products are frequently bought together to boost your sales!")
    
    # Sidebar controls
    st.sidebar.subheader("🎛️ Analysis Settings")
    min_support = st.sidebar.slider("Minimum Support", 0.001, 0.1, 0.01, 0.001)
    min_confidence = st.sidebar.slider("Minimum Confidence", 0.1, 0.9, 0.1, 0.05)
    min_lift = st.sidebar.slider("Minimum Lift", 1.0, 5.0, 1.0, 0.1)
    
    # Phase 4.1: Advanced optimization controls
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚡ Performance Settings")
    
    with st.sidebar.expander("🔧 Advanced Options"):
        max_transactions = st.slider("Max Transactions", 1000, 50000, 10000, 1000,
                                   help="Limit transactions for faster analysis")
        max_products = st.slider("Max Products", 50, 1000, 200, 50,
                                help="Analyze only top N most frequent products")
        min_basket_size = st.slider("Min Basket Size", 1, 10, 2, 1,
                                   help="Minimum items per transaction")
        
        if st.button("🔄 Apply Optimization Settings"):
            st.session_state.optimization_updated = True
    
    # Initialize market basket analyzer with caching and custom optimization settings
    analyzer = MarketBasketAnalyzer(min_support, min_confidence, min_lift)
    
    # Apply custom optimization settings if provided
    if 'optimization_updated' in st.session_state or True:  # Always apply for now
        analyzer.update_optimization_settings(
            max_transactions=max_transactions,
            max_products=max_products,
            min_transaction_items=min_basket_size,
            max_transaction_items=50  # Keep reasonable upper limit
        )
    
    # Phase 4.1: Progressive loading with better UX
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    
    with st.spinner("🔄 Running Market Basket Analysis..."):
        # Show optimization settings
        with st.expander("⚙️ Optimization Settings", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Max Transactions", analyzer.max_transactions)
                st.metric("Max Products", analyzer.max_products)
            with col2:
                st.metric("Min Items/Transaction", analyzer.min_transaction_items)
                st.metric("Max Items/Transaction", analyzer.max_transaction_items)
        
        # Run analysis with progress tracking
        frequent_itemsets, rules, from_cache = analyzer.run_cached_analysis(data)
        
        if from_cache:
            st.success("✅ Loaded cached result - Analysis completed instantly!")
        else:
            st.success("✅ Analysis completed with Phase 4.1 optimizations!")
            
            # Show performance summary
            summary = analyzer.get_analysis_summary()
            if summary:
                st.info(f"📊 Processed {summary['total_transactions']} transactions with {summary['unique_products']} products")
    
    if frequent_itemsets is None or len(frequent_itemsets) == 0:
        st.error("Failed to generate frequent itemsets. Try lowering the minimum support.")
        return
    
    # Product selection
    st.subheader("🔍 Product Recommendations")
    selected_product = st.selectbox(
        "Select a product to get cross-selling recommendations:",
        options=products['Description'].tolist(),
        index=0
    )
    
    if selected_product:
        # Get recommendations
        recommendations = analyzer.get_product_recommendations(selected_product, top_n=5)
        
        if recommendations:
            st.success(f"✅ Found {len(recommendations)} recommendations for '{selected_product}'")
            
            # Display recommendations
            for i, rec in enumerate(recommendations, 1):
                with st.expander(f"#{i} {rec['recommended_product']}", expanded=i==1):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Confidence", f"{rec['confidence']:.1%}")
                    with col2:
                        st.metric("Lift", f"{rec['lift']:.2f}")
                    with col3:
                        st.metric("Support", f"{rec['support']:.1%}")
                    
                    st.info(rec['explanation'])
            
            # Visualization
            if len(recommendations) > 0:
                st.subheader("📊 Recommendation Metrics")
                rec_df = pd.DataFrame(recommendations)
                
                fig = make_subplots(
                    rows=1, cols=3,
                    subplot_titles=('Confidence', 'Lift', 'Support'),
                    specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
                )
                
                # Confidence chart
                fig.add_trace(
                    go.Bar(x=rec_df['recommended_product'], y=rec_df['confidence'], 
                           name='Confidence', marker_color='#1f77b4'),
                    row=1, col=1
                )
                
                # Lift chart
                fig.add_trace(
                    go.Bar(x=rec_df['recommended_product'], y=rec_df['lift'], 
                           name='Lift', marker_color='#ff7f0e'),
                    row=1, col=2
                )
                
                # Support chart
                fig.add_trace(
                    go.Bar(x=rec_df['recommended_product'], y=rec_df['support'], 
                           name='Support', marker_color='#2ca02c'),
                    row=1, col=3
                )
                
                fig.update_layout(height=400, showlegend=False)
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.warning(f"⚠️ No recommendations found for '{selected_product}'. Try adjusting the analysis settings.")
    
    # Top association rules
    st.subheader("🏆 Top Association Rules")
    
    if rules is not None and len(rules) > 0:
        # Display top rules directly from the rules DataFrame
        display_rules = rules.head(10).copy()
        display_rules = display_rules[['antecedents_str', 'consequents_str', 'confidence', 'lift', 'support']]
        display_rules.columns = ['Antecedent (If)', 'Consequent (Then)', 'Confidence', 'Lift', 'Support']
        
        # Format the values for better display
        display_rules['Confidence'] = display_rules['Confidence'].round(3)
        display_rules['Lift'] = display_rules['Lift'].round(3)
        display_rules['Support'] = display_rules['Support'].round(3)
        
        st.dataframe(display_rules, use_container_width=True)
        
        # Show rule interpretation
        st.info("💡 **How to read these rules:** If a customer buys the Antecedent product(s), they are likely to also buy the Consequent product(s). Higher Confidence, Lift, and Support indicate stronger associations.")
    else:
        st.warning("⚠️ No association rules found with current settings. Try lowering the minimum support, confidence, or lift thresholds.")
    
    # Analysis summary
    summary = analyzer.get_analysis_summary()
    if summary:
        st.subheader("📋 Analysis Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Transactions", summary.get('total_transactions', 0))
        with col2:
            st.metric("Unique Products", summary.get('unique_products', 0))
        with col3:
            st.metric("Association Rules", summary.get('association_rules_count', 0))

def show_forecasting_page(data, products):
    """Display inventory forecasting page"""
    st.header("📈 Inventory Forecaster")
    st.write("Predict future demand to optimize your inventory levels!")
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["🎯 Individual Product Forecast", "🏆 Top Forecasted Products"])
    
    with tab1:
        # Product selection
        st.subheader("🎯 Select Product for Forecasting")
        selected_product_id = st.selectbox(
            "Choose a product:",
            options=products['ProductID'].tolist(),
            format_func=lambda x: f"{x} - {products[products['ProductID']==x]['Description'].iloc[0]}"
        )
        
        # Forecast parameters
        col1, col2 = st.columns(2)
        with col1:
            forecast_days = st.slider("Forecast Period (days)", 7, 90, 30)
        with col2:
            history_days = st.slider("Historical Data to Show (days)", 30, 180, 60)
        
        if st.button("🔮 Generate Forecast", type="primary"):
            # Initialize forecaster with caching
            forecaster = InventoryForecaster()
            
            with st.spinner("🔄 Running ARIMA forecasting..."):
                # Use cached forecasting
                forecast_result, from_cache = forecaster.run_cached_forecast(
                    data, selected_product_id, forecast_days
                )
                
                if from_cache:
                    st.info("✅ Loaded cached result")
                else:
                    st.warning("⚠️ Recomputing model... Please wait.")
                
                if forecast_result:
                    st.success("✅ Forecast generated successfully!")
                    
                    # Get summary and chart data
                    summary = forecaster.get_forecast_summary(selected_product_id)
                    chart_data = forecaster.get_forecast_chart_data(selected_product_id, history_days)
                    
                    # Display key metrics
                    st.subheader("📊 Forecast Summary")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Avg Historical Demand", f"{summary['avg_historical_demand']:.1f}")
                    with col2:
                        st.metric("Avg Forecast Demand", f"{summary['avg_forecast_demand']:.1f}")
                    with col3:
                        st.metric("Total Forecast Demand", f"{summary['total_forecast_demand']:.0f}")
                    with col4:
                        st.metric("Recommended Stock", f"{summary['recommended_stock_level']:.0f}")
                    
                    # Trend indicator
                    trend_color = "🟢" if summary['trend'] == "increasing" else "🔴"
                    st.markdown(f"**Trend**: {trend_color} {summary['trend'].title()} ({summary['change_percent']:+.1f}%)")
                    
                    # Forecast chart
                    st.subheader("📈 Demand Forecast Chart")
                    
                    if chart_data:
                        fig = go.Figure()
                        
                        # Historical data
                        fig.add_trace(go.Scatter(
                            x=chart_data['historical_dates'],
                            y=chart_data['historical_values'],
                            mode='lines+markers',
                            name='Historical Demand',
                            line=dict(color='#1f77b4')
                        ))
                        
                        # Forecast data
                        fig.add_trace(go.Scatter(
                            x=chart_data['forecast_dates'],
                            y=chart_data['forecast_values'],
                            mode='lines+markers',
                            name='Forecast',
                            line=dict(color='#ff7f0e', dash='dash')
                        ))
                        
                        # Confidence interval
                        fig.add_trace(go.Scatter(
                            x=chart_data['forecast_dates'] + chart_data['forecast_dates'][::-1],
                            y=chart_data['forecast_upper'] + chart_data['forecast_lower'][::-1],
                            fill='toself',
                            fillcolor='rgba(255,127,14,0.2)',
                            line=dict(color='rgba(255,255,255,0)'),
                            name='Confidence Interval'
                        ))
                        
                        fig.update_layout(
                            title=f"Demand Forecast for {selected_product_id}",
                            xaxis_title="Date",
                            yaxis_title="Quantity",
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Insights
                    st.subheader("💡 Actionable Insights")
                    for insight in summary['insights']:
                        st.info(insight)
                    
                    # Model accuracy
                    if summary.get('model_accuracy'):
                        st.subheader("🎯 Model Accuracy")
                        accuracy_color = "🟢" if summary['model_accuracy'] < 20 else "🟡" if summary['model_accuracy'] < 40 else "🔴"
                        st.markdown(f"**MAPE**: {accuracy_color} {summary['model_accuracy']:.1f}%")
                        
                        if summary['model_accuracy'] < 20:
                            st.success("Excellent forecast accuracy!")
                        elif summary['model_accuracy'] < 40:
                            st.warning("Moderate forecast accuracy. Consider using additional data.")
                        else:
                            st.error("Low forecast accuracy. Results should be used with caution.")
                
                else:
                    st.error("❌ Failed to generate forecast. This product may have insufficient data.")
    
    with tab2:
        # Top forecasted products for restocking
        st.subheader("🏆 Top Products for Restocking (Next 30 Days)")
        st.write("Products with highest forecasted demand - prioritize these for inventory restocking!")
        
        # Parameters for top products analysis
        col1, col2 = st.columns(2)
        with col1:
            top_n = st.slider("Number of products to analyze", 5, 20, 10)
        with col2:
            forecast_period = st.slider("Forecast period (days)", 7, 60, 30)
        
        if st.button("📊 Generate Top Products Forecast", type="primary"):
            forecaster = InventoryForecaster()
            
            with st.spinner(f"🔄 Analyzing top {top_n} products for demand forecasting..."):
                top_products_df = forecaster.get_top_forecasted_products(
                    data, top_n=top_n, forecast_days=forecast_period
                )
                
                if not top_products_df.empty:
                    st.success(f"✅ Generated forecasts for {len(top_products_df)} products!")
                    
                    # Display the results table
                    st.dataframe(
                        top_products_df[['ProductID', 'Description', 'Total_Forecasted_Demand', 
                                       'Avg_Daily_Demand', 'Peak_Demand', 'Priority', 'Recommendation']],
                        use_container_width=True,
                        column_config={
                            "ProductID": "Product ID",
                            "Description": "Product Name",
                            "Total_Forecasted_Demand": st.column_config.NumberColumn(
                                "Total Demand",
                                help="Total forecasted demand for the period",
                                format="%.1f"
                            ),
                            "Avg_Daily_Demand": st.column_config.NumberColumn(
                                "Daily Avg",
                                help="Average daily demand",
                                format="%.1f"
                            ),
                            "Peak_Demand": st.column_config.NumberColumn(
                                "Peak Demand",
                                help="Highest single day demand",
                                format="%.1f"
                            ),
                            "Priority": st.column_config.TextColumn(
                                "Priority",
                                help="Restocking priority level"
                            ),
                            "Recommendation": st.column_config.TextColumn(
                                "Recommendation",
                                help="Actionable restocking recommendation"
                            )
                        }
                    )
                    
                    # Summary metrics
                    st.subheader("📋 Restocking Summary")
                    high_priority = len(top_products_df[top_products_df['Priority'] == 'High'])
                    medium_priority = len(top_products_df[top_products_df['Priority'] == 'Medium'])
                    total_demand = top_products_df['Total_Forecasted_Demand'].sum()
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("🔴 High Priority", high_priority)
                    with col2:
                        st.metric("🟡 Medium Priority", medium_priority)
                    with col3:
                        st.metric("📦 Total Forecasted Demand", f"{total_demand:.0f}")
                    with col4:
                        st.metric("📊 Average Daily Demand", f"{top_products_df['Avg_Daily_Demand'].mean():.1f}")
                    
                    # Priority breakdown chart
                    priority_counts = top_products_df['Priority'].value_counts()
                    if len(priority_counts) > 0:
                        fig = px.pie(
                            values=priority_counts.values,
                            names=priority_counts.index,
                            title="Restocking Priority Distribution",
                            color_discrete_map={
                                'High': '#ff4444',
                                'Medium': '#ffaa00', 
                                'Low': '#44ff44',
                                'Monitor': '#cccccc'
                            }
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    st.error("❌ Failed to generate forecasts. Please check your data quality.")
                    
        # Clean up memory after analysis
        cleanup_memory()

def show_upload_page():
    """Display data upload page"""
    st.header("📤 Upload Your Data")
    st.write("Upload your own sales data to get personalized insights!")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file with your sales transaction data"
    )
    
    # Sample data download
    st.subheader("📄 Sample Data Format")
    st.write("Your CSV file should have the following columns:")
    
    sample_columns = ['InvoiceNo', 'StockCode', 'Description', 'Quantity', 'InvoiceDate', 'UnitPrice', 'CustomerID']
    sample_df = pd.DataFrame({col: [f"Sample {col}"] for col in sample_columns})
    st.dataframe(sample_df, use_container_width=True)
    
    # Download sample
    if st.button("📥 Download Sample CSV"):
        from utils.data_processing import create_sample_csv
        sample_data = create_sample_csv()
        csv = sample_data.to_csv(index=False)
        st.download_button(
            label="Download sample_data.csv",
            data=csv,
            file_name="sample_data.csv",
            mime="text/csv"
        )
    
    # Process uploaded file
    if uploaded_file is not None:
        try:
            # Read uploaded file
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded successfully! {len(df)} records found.")
            
            # Validate data
            processor = DataProcessor()
            is_valid, validation_messages = processor.validate_upload_data(df)
            
            if is_valid:
                st.success("✅ Data validation passed!")
                
                # Show data preview
                st.subheader("👀 Data Preview")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Process and analyze
                if st.button("🚀 Process and Analyze Data", type="primary"):
                    with st.spinner("🔄 Processing your data..."):
                        # Clean data
                        clean_df = processor.clean_retail_data(df)
                        
                        if len(clean_df) > 0:
                            st.success(f"✅ Data processed successfully! {len(clean_df)} clean records.")
                            
                            # Store in session state for analysis
                            st.session_state['uploaded_data'] = clean_df
                            st.session_state['uploaded_products'] = clean_df[['StockCode', 'Description']].drop_duplicates()
                            
                            st.info("🎯 Data is ready! Navigate to Cross-Selling or Forecasting pages to analyze your data.")
                        else:
                            st.error("❌ No valid data remaining after cleaning.")
            else:
                st.error("❌ Data validation failed:")
                for message in validation_messages:
                    st.error(f"• {message}")
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

if __name__ == "__main__":
    main()
