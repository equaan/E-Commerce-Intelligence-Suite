"""
Inventory Forecasting using ARIMA Time Series Analysis
Predicts future demand for products to optimize inventory management
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
import sys
import os
from typing import Dict, Any, Tuple, Optional

# Add utils to path for cache manager
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from cache_manager import load_result, save_result, cleanup_memory

warnings.filterwarnings('ignore')

class InventoryForecaster:
    def __init__(self):
        """Initialize Inventory Forecaster with Phase 5.1 optimizations"""
        self.models = {}
        self.forecasts = {}
        self.product_data = {}
        
        # Phase 5.1 optimization settings
        self.max_time_series_length = 365  # Limit history for faster processing
        self.high_volume_threshold = 1000   # Use ARIMA for high-volume products
        self.complexity_threshold = 100     # Use simple models for low-complexity data
        self.enable_parallel_processing = True
        self.max_workers = 4
    
    def prepare_product_data(self, df, product_id):
        """
        Prepare time series data for a specific product
        
        Args:
            df: DataFrame with columns ['Date', 'ProductID'/'StockCode', 'Quantity']
            product_id: Product ID to forecast
        """
        # Handle both ProductID and StockCode columns
        product_col = 'StockCode' if 'StockCode' in df.columns else 'ProductID'
        
        # Filter data for specific product
        product_data = df[df[product_col] == product_id].copy()
        
        if len(product_data) == 0:
            return None
        
        # Handle different date column names
        date_col = 'Date' if 'Date' in product_data.columns else 'DateID'
        
        # Convert date column to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(product_data[date_col]):
            product_data[date_col] = pd.to_datetime(product_data[date_col])
        
        # Sort by date and aggregate duplicates (sum quantities for same date)
        product_data = product_data.groupby(date_col)['Quantity'].sum().to_frame()
        
        # Phase 5.1 Optimization: Limit time series length for faster processing
        product_data = self._optimize_time_series_length(product_data)
        
        # Ensure daily frequency and fill missing dates
        product_data = product_data.asfreq('D', fill_value=0)
        
        # Store product data
        self.product_data[product_id] = product_data
        
        return product_data
    
    def _optimize_time_series_length(self, data):
        """Phase 5.1 Optimization: Limit time series length for faster processing"""
        if len(data) <= self.max_time_series_length:
            return data
        
        # Keep recent data + sample from historical data
        recent_points = self.max_time_series_length // 2
        historical_points = self.max_time_series_length - recent_points
        
        recent_data = data.tail(recent_points)
        historical_data = data.head(-recent_points)
        
        if len(historical_data) > historical_points:
            # Sample from historical data to maintain temporal distribution
            sample_indices = np.linspace(0, len(historical_data)-1, historical_points, dtype=int)
            sampled_historical = historical_data.iloc[sample_indices]
        else:
            sampled_historical = historical_data
        
        optimized_data = pd.concat([sampled_historical, recent_data]).sort_index()
        
        return optimized_data
    
    def _classify_product_complexity(self, product_data):
        """Phase 5.1: Classify product forecasting complexity"""
        if len(product_data) < 30:
            return "insufficient_data"
        
        total_volume = product_data['Quantity'].sum()
        variance = product_data['Quantity'].var()
        mean_value = product_data['Quantity'].mean()
        
        # High volume products get ARIMA for best accuracy
        if total_volume > self.high_volume_threshold:
            return "high_volume"
        
        # Check for trend
        if self._has_trend(product_data):
            return "trending"
        
        # Check for seasonality
        if self._has_seasonality(product_data):
            return "seasonal"
        
        # Low complexity - use simple models
        if variance < self.complexity_threshold or mean_value < 1:
            return "simple"
        
        return "medium"
    
    def _has_trend(self, data):
        """Detect if data has a significant trend"""
        try:
            # Simple trend detection using correlation with time
            x = np.arange(len(data))
            correlation = np.corrcoef(x, data['Quantity'].values)[0, 1]
            return abs(correlation) > 0.3
        except:
            return False
    
    def _has_seasonality(self, data):
        """Detect if data has seasonality (requires at least 2 cycles)"""
        try:
            if len(data) < 60:  # Need at least 2 months for monthly seasonality
                return False
            
            # Simple seasonality detection using autocorrelation
            autocorr_7 = data['Quantity'].autocorr(lag=7)   # Weekly
            autocorr_30 = data['Quantity'].autocorr(lag=30) # Monthly
            
            return max(abs(autocorr_7), abs(autocorr_30)) > 0.3
        except:
            return False
    
    def _choose_optimal_model(self, product_data, product_id):
        """Phase 5.1: Choose optimal forecasting model based on data characteristics"""
        complexity = self._classify_product_complexity(product_data)
        
        model_choice = {
            "insufficient_data": "moving_average",
            "simple": "moving_average", 
            "trending": "linear_trend",
            "seasonal": "exponential_smoothing",
            "medium": "exponential_smoothing",
            "high_volume": "arima"
        }
        
        selected_model = model_choice.get(complexity, "exponential_smoothing")
        
        return selected_model
    
    def _forecast_moving_average(self, data, forecast_days=30):
        """Fast forecasting using moving average (1000x faster than ARIMA)"""
        window = min(7, len(data) // 2)  # Use 7-day or half the data length
        if window < 1:
            window = 1
        
        moving_avg = data['Quantity'].rolling(window=window).mean().iloc[-1]
        if pd.isna(moving_avg):
            moving_avg = data['Quantity'].mean()
        
        forecast = pd.Series([moving_avg] * forecast_days)
        return {
            'forecast': forecast,
            'model_type': 'moving_average',
            'window': window,
            'confidence_intervals': None
        }
    
    def _forecast_linear_trend(self, data, forecast_days=30):
        """Fast forecasting using linear trend (100x faster than ARIMA)"""
        try:
            x = np.arange(len(data))
            y = data['Quantity'].values
            
            # Simple linear regression
            slope, intercept = np.polyfit(x, y, 1)
            
            # Generate forecast
            future_x = np.arange(len(data), len(data) + forecast_days)
            forecast = slope * future_x + intercept
            
            # Ensure non-negative forecasts
            forecast = np.maximum(forecast, 0)
            
            return {
                'forecast': pd.Series(forecast),
                'model_type': 'linear_trend',
                'slope': slope,
                'intercept': intercept,
                'confidence_intervals': None
            }
        except:
            # Fallback to moving average
            return self._forecast_moving_average(data, forecast_days)
    
    def _forecast_exponential_smoothing(self, data, forecast_days=30):
        """Medium-speed forecasting using Exponential Smoothing (10-50x faster than ARIMA)"""
        try:
            # Prepare data
            ts_data = data['Quantity'].dropna()
            if len(ts_data) < 10:
                return self._forecast_moving_average(data, forecast_days)
            
            # Fit exponential smoothing model
            model = ExponentialSmoothing(
                ts_data,
                trend='add' if self._has_trend(data) else None,
                seasonal='add' if self._has_seasonality(data) and len(ts_data) > 24 else None,
                seasonal_periods=7 if len(ts_data) > 24 else None
            )
            
            fitted_model = model.fit(optimized=True, remove_bias=True)
            forecast = fitted_model.forecast(steps=forecast_days)
            
            # Ensure non-negative forecasts
            forecast = np.maximum(forecast, 0)
            
            return {
                'forecast': forecast,
                'model_type': 'exponential_smoothing',
                'aic': fitted_model.aic if hasattr(fitted_model, 'aic') else None,
                'confidence_intervals': None
            }
        except Exception as e:
            print(f"   Exponential smoothing failed: {str(e)}, using linear trend")
            return self._forecast_linear_trend(data, forecast_days)
    
    def _forecast_optimized(self, product_data, product_id, forecast_days=30):
        """Phase 5.1: Use optimal model based on data characteristics"""
        model_type = self._choose_optimal_model(product_data, product_id)
        
        if model_type == "arima":
            return self.train_arima_model(product_id, forecast_days)
        elif model_type == "exponential_smoothing":
            return self._forecast_exponential_smoothing(product_data, forecast_days)
        elif model_type == "linear_trend":
            return self._forecast_linear_trend(product_data, forecast_days)
        else:  # moving_average
            return self._forecast_moving_average(product_data, forecast_days)
    
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
    
    def run_cached_forecast(self, df, product_id, forecast_days=30, data_hash=None):
        """
        Run ARIMA forecasting with caching support
        
        Args:
            df: DataFrame with sales data
            product_id: Product ID to forecast
            forecast_days: Number of days to forecast
            data_hash: Optional hash of the data for cache key
            
        Returns:
            Tuple of (forecast_result, from_cache)
        """
        # Prepare data first
        product_data = self.prepare_product_data(df, product_id)
        if product_data is None:
            return None, False
        
        # Create cache parameters
        cache_params = {
            'product_id': product_id,
            'forecast_days': forecast_days,
            'data_shape': product_data.shape,
            'data_hash': data_hash or str(hash(str(product_data.values.tobytes()))),
            'data_start': str(product_data.index.min()),
            'data_end': str(product_data.index.max()),
            'total_quantity': float(product_data['Quantity'].sum())
        }
        
        # Try to load from cache
        cached_result = load_result("arima_results", cache_params)
        if cached_result is not None:
            self.forecasts[product_id] = cached_result
            return cached_result, True
        
        # Phase 5.1: Use optimized forecasting instead of always ARIMA
        forecast_result = self._forecast_optimized(product_data, product_id, forecast_days)
        
        if forecast_result is not None:
            # Save to cache
            save_result("arima_results", cache_params, forecast_result)
            
            # Clean up memory
            cleanup_memory()
        
        return forecast_result, False

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
    
    def get_top_forecasted_products(self, df, top_n=10, forecast_days=30):
        """
        Get top products by forecasted demand with Phase 5.1 parallel processing
        
        Args:
            df: DataFrame with sales data
            top_n: Number of top products to return
            forecast_days: Number of days to forecast
            
        Returns:
            DataFrame with top forecasted products and recommendations
        """
        # Get top products by historical sales - handle both StockCode and ProductID
        product_col = 'StockCode' if 'StockCode' in df.columns else 'ProductID'
        top_products = (df.groupby([product_col, 'Description'])['Quantity']
                       .sum()
                       .sort_values(ascending=False)
                       .head(top_n * 2)  # Get more to account for failed forecasts
                       .index.tolist())
        
        # Phase 5.1: Use parallel processing for bulk forecasting
        if self.enable_parallel_processing and len(top_products) > 2:
            forecast_results = self._forecast_products_parallel(df, top_products, forecast_days, top_n)
        else:
            forecast_results = self._forecast_products_sequential(df, top_products, forecast_days, top_n)
        
        # Convert to DataFrame and sort by total forecasted demand
        if forecast_results:
            results_df = pd.DataFrame(forecast_results)
            results_df = results_df.sort_values('Total_Forecasted_Demand', ascending=False)
            return results_df
        else:
            return pd.DataFrame()
    
    def _forecast_products_sequential(self, df, top_products, forecast_days, top_n):
        """Sequential forecasting (original method)"""
        forecast_results = []
        
        for i, (stock_code, description) in enumerate(top_products):
            try:
                # Run cached forecast
                forecast_result, from_cache = self.run_cached_forecast(
                    df, stock_code, forecast_days
                )
                
                if forecast_result is not None:
                    result = self._process_forecast_result(
                        stock_code, description, forecast_result, forecast_days
                    )
                    if result:
                        forecast_results.append(result)
                    
                    if len(forecast_results) >= top_n:
                        break
                        
            except Exception as e:
                continue
        
        return forecast_results
    
    def _forecast_products_parallel(self, df, top_products, forecast_days, top_n):
        """Phase 5.1: Parallel forecasting using ProcessPoolExecutor"""
        forecast_results = []
        
        # Create tasks for parallel processing
        tasks = [(df, stock_code, description, forecast_days) for stock_code, description in top_products[:top_n*2]]
        
        try:
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_product = {
                    executor.submit(self._forecast_single_product_worker, task): task[1] 
                    for task in tasks
                }
                
                # Collect results as they complete
                for i, future in enumerate(as_completed(future_to_product)):
                    try:
                        stock_code = future_to_product[future]
                        result = future.result(timeout=30)  # 30 second timeout per product
                        
                        if result:
                            forecast_results.append(result)
                        
                        if len(forecast_results) >= top_n:
                            break
                            
                    except Exception as e:
                        continue
                        
        except Exception as e:
            return self._forecast_products_sequential(df, top_products, forecast_days, top_n)
        
        return forecast_results
    
    def _forecast_single_product_worker(self, task):
        """Worker function for parallel processing"""
        df, stock_code, description, forecast_days = task
        
        try:
            # Create a new forecaster instance for this worker
            forecaster = InventoryForecaster()
            
            # Run forecast
            forecast_result, from_cache = forecaster.run_cached_forecast(
                df, stock_code, forecast_days
            )
            
            if forecast_result is not None:
                return self._process_forecast_result(
                    stock_code, description, forecast_result, forecast_days
                )
        except Exception as e:
            return None
        
        return None
    
    def _process_forecast_result(self, stock_code, description, forecast_result, forecast_days):
        """Process forecast result into standardized format"""
        try:
            forecast = forecast_result['forecast']
            
            # Calculate metrics
            total_forecasted_demand = forecast.sum()
            avg_daily_demand = forecast.mean()
            peak_demand = forecast.max()
            demand_variability = forecast.std() / forecast.mean() if forecast.mean() > 0 else 0
            
            # Generate recommendation
            if avg_daily_demand > 5:
                if demand_variability > 0.3:
                    recommendation = "🔴 High Priority - High demand with variability"
                    priority = "High"
                else:
                    recommendation = "🟡 Medium Priority - Steady high demand"
                    priority = "Medium"
            elif avg_daily_demand > 2:
                recommendation = "🟢 Low Priority - Moderate demand"
                priority = "Low"
            else:
                recommendation = "⚪ Monitor - Low demand"
                priority = "Monitor"
            
            return {
                'ProductID': stock_code,
                'Description': description,
                'Total_Forecasted_Demand': round(total_forecasted_demand, 1),
                'Avg_Daily_Demand': round(avg_daily_demand, 1),
                'Peak_Demand': round(peak_demand, 1),
                'Demand_Variability': round(demand_variability, 2),
                'Priority': priority,
                'Recommendation': recommendation,
                'Forecast_Period_Days': forecast_days,
                'Model_Type': forecast_result.get('model_type', 'unknown')
            }
        except Exception as e:
            return None

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
