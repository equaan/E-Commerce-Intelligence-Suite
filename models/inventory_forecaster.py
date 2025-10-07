"""
Inventory Forecasting using ARIMA Time Series Analysis
Predicts future demand for products to optimize inventory management
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')

class InventoryForecaster:
    def __init__(self):
        """Initialize Inventory Forecaster"""
        self.models = {}
        self.forecasts = {}
        self.product_data = {}
    
    def prepare_product_data(self, df, product_id):
        """
        Prepare time series data for a specific product
        
        Args:
            df: DataFrame with columns ['Date', 'StockCode', 'Quantity']
            product_id: Product ID to forecast
        """
        # Filter data for specific product
        product_data = df[df['StockCode'] == product_id].copy()
        
        if len(product_data) == 0:
            return None
        
        # Sort by date and set as index
        product_data = product_data.sort_values('Date')
        product_data.set_index('Date', inplace=True)
        
        # Ensure daily frequency and fill missing dates
        product_data = product_data.asfreq('D', fill_value=0)
        
        # Store product data
        self.product_data[product_id] = product_data
        
        return product_data
    
    def check_stationarity(self, series):
        """Check if time series is stationary using Augmented Dickey-Fuller test"""
        try:
            result = adfuller(series.dropna())
            p_value = result[1]
            is_stationary = p_value < 0.05
            
            return {
                'is_stationary': is_stationary,
                'p_value': p_value,
                'critical_values': result[4]
            }
        except:
            return {'is_stationary': False, 'p_value': 1.0, 'critical_values': {}}
    
    def find_optimal_arima_params(self, series, max_p=3, max_d=2, max_q=3):
        """
        Find optimal ARIMA parameters using AIC criterion
        
        Args:
            series: Time series data
            max_p, max_d, max_q: Maximum values for ARIMA parameters
        """
        best_aic = float('inf')
        best_params = (1, 1, 1)
        
        # Try different parameter combinations
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        model = ARIMA(series, order=(p, d, q))
                        fitted_model = model.fit()
                        aic = fitted_model.aic
                        
                        if aic < best_aic:
                            best_aic = aic
                            best_params = (p, d, q)
                    except:
                        continue
        
        return best_params, best_aic
    
    def train_arima_model(self, product_id, forecast_days=30):
        """
        Train ARIMA model for a specific product
        
        Args:
            product_id: Product ID to train model for
            forecast_days: Number of days to forecast
        """
        if product_id not in self.product_data:
            return None
        
        data = self.product_data[product_id]['Quantity']
        
        # Check if we have enough data
        if len(data) < 30:
            print(f"⚠️  Insufficient data for {product_id} (need at least 30 days)")
            return None
        
        try:
            # Find optimal parameters
            optimal_params, aic = self.find_optimal_arima_params(data)
            print(f"📊 Optimal ARIMA parameters for {product_id}: {optimal_params} (AIC: {aic:.2f})")
            
            # Train model
            model = ARIMA(data, order=optimal_params)
            fitted_model = model.fit()
            
            # Generate forecast
            forecast = fitted_model.forecast(steps=forecast_days)
            forecast_ci = fitted_model.get_forecast(steps=forecast_days).conf_int()
            
            # Create forecast dates
            last_date = data.index[-1]
            forecast_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=forecast_days,
                freq='D'
            )
            
            # Store results
            self.models[product_id] = fitted_model
            self.forecasts[product_id] = {
                'forecast': forecast,
                'forecast_dates': forecast_dates,
                'confidence_interval': forecast_ci,
                'model_params': optimal_params,
                'aic': aic,
                'historical_data': data
            }
            
            return self.forecasts[product_id]
            
        except Exception as e:
            print(f"❌ Error training model for {product_id}: {str(e)}")
            return None
    
    def get_forecast_summary(self, product_id):
        """Get forecast summary with actionable insights"""
        if product_id not in self.forecasts:
            return None
        
        forecast_data = self.forecasts[product_id]
        forecast = forecast_data['forecast']
        historical = forecast_data['historical_data']
        
        # Calculate summary statistics
        avg_historical_demand = historical.mean()
        avg_forecast_demand = forecast.mean()
        total_forecast_demand = forecast.sum()
        
        # Generate insights
        trend = "increasing" if avg_forecast_demand > avg_historical_demand else "decreasing"
        change_percent = ((avg_forecast_demand - avg_historical_demand) / avg_historical_demand) * 100
        
        # Stock recommendations
        safety_stock = avg_forecast_demand * 0.2  # 20% safety stock
        recommended_stock = total_forecast_demand + safety_stock
        
        summary = {
            'product_id': product_id,
            'forecast_period_days': len(forecast),
            'avg_historical_demand': round(avg_historical_demand, 2),
            'avg_forecast_demand': round(avg_forecast_demand, 2),
            'total_forecast_demand': round(total_forecast_demand, 2),
            'trend': trend,
            'change_percent': round(change_percent, 1),
            'recommended_stock_level': round(recommended_stock, 0),
            'safety_stock': round(safety_stock, 0),
            'model_accuracy': self._calculate_model_accuracy(product_id),
            'insights': self._generate_insights(product_id, trend, change_percent)
        }
        
        return summary
    
    def _calculate_model_accuracy(self, product_id):
        """Calculate model accuracy using MAPE (Mean Absolute Percentage Error)"""
        if product_id not in self.models:
            return None
        
        try:
            model = self.models[product_id]
            historical = self.forecasts[product_id]['historical_data']
            
            # Use last 30 days for validation
            if len(historical) < 60:
                return None
            
            train_data = historical[:-30]
            test_data = historical[-30:]
            
            # Retrain on training data
            temp_model = ARIMA(train_data, order=self.forecasts[product_id]['model_params'])
            temp_fitted = temp_model.fit()
            
            # Forecast test period
            test_forecast = temp_fitted.forecast(steps=30)
            
            # Calculate MAPE
            mape = np.mean(np.abs((test_data - test_forecast) / test_data)) * 100
            
            return round(mape, 2)
            
        except:
            return None
    
    def _generate_insights(self, product_id, trend, change_percent):
        """Generate actionable insights based on forecast"""
        insights = []
        
        if abs(change_percent) < 5:
            insights.append("📊 Demand is expected to remain stable")
        elif trend == "increasing":
            if change_percent > 20:
                insights.append("📈 Strong demand growth expected - consider increasing inventory")
            else:
                insights.append("📈 Moderate demand increase expected")
        else:
            if change_percent < -20:
                insights.append("📉 Significant demand decline expected - reduce inventory")
            else:
                insights.append("📉 Slight demand decrease expected")
        
        # Seasonal insights (simplified)
        forecast_data = self.forecasts[product_id]
        forecast = forecast_data['forecast']
        
        if forecast.std() > forecast.mean() * 0.3:
            insights.append("⚡ High demand variability - maintain higher safety stock")
        
        return insights
    
    def get_forecast_chart_data(self, product_id, days_history=60):
        """Get data formatted for plotting forecast charts"""
        if product_id not in self.forecasts:
            return None
        
        forecast_data = self.forecasts[product_id]
        historical = forecast_data['historical_data']
        forecast = forecast_data['forecast']
        forecast_dates = forecast_data['forecast_dates']
        confidence_interval = forecast_data['confidence_interval']
        
        # Get recent historical data
        recent_historical = historical.tail(days_history)
        
        chart_data = {
            'historical_dates': recent_historical.index.tolist(),
            'historical_values': recent_historical.values.tolist(),
            'forecast_dates': forecast_dates.tolist(),
            'forecast_values': forecast.tolist(),
            'forecast_lower': confidence_interval.iloc[:, 0].tolist(),
            'forecast_upper': confidence_interval.iloc[:, 1].tolist()
        }
        
        return chart_data
    
    def forecast_multiple_products(self, df, product_list, forecast_days=30):
        """Forecast demand for multiple products"""
        results = {}
        
        print(f"🔮 Forecasting demand for {len(product_list)} products...")
        
        for i, product_id in enumerate(product_list):
            print(f"   Processing {i+1}/{len(product_list)}: {product_id}")
            
            # Prepare data
            self.prepare_product_data(df, product_id)
            
            # Train model and forecast
            forecast_result = self.train_arima_model(product_id, forecast_days)
            
            if forecast_result:
                summary = self.get_forecast_summary(product_id)
                results[product_id] = summary
        
        print(f"✅ Completed forecasting for {len(results)} products")
        return results

if __name__ == "__main__":
    # Test inventory forecaster
    print("Testing Inventory Forecaster...")
    
    # Create sample time series data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    sample_data = pd.DataFrame({
        'Date': dates,
        'StockCode': ['PROD001'] * 100,
        'Quantity': np.random.poisson(10, 100) + np.sin(np.arange(100) * 0.1) * 3
    })
    
    forecaster = InventoryForecaster()
    forecaster.prepare_product_data(sample_data, 'PROD001')
    result = forecaster.train_arima_model('PROD001', forecast_days=14)
    
    if result:
        summary = forecaster.get_forecast_summary('PROD001')
        print("Forecast completed!")
        print(f"Average forecast demand: {summary['avg_forecast_demand']}")
    else:
        print("Forecast failed!")
