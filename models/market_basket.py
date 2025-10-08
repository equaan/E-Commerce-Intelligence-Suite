"""
Market Basket Analysis using Apriori Algorithm - OPTIMIZED VERSION
Implements cross-selling recommendations with Phase 4.1 performance optimizations
"""

import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import warnings
import sys
import os

# Add utils to path for cache manager
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from cache_manager import load_result, save_result, cleanup_memory

warnings.filterwarnings('ignore')

class MarketBasketAnalyzer:
    def __init__(self, min_support=0.01, min_confidence=0.1, min_lift=1.0):
        """
        Initialize Market Basket Analyzer with optimizations
        
        Args:
            min_support: Minimum support threshold for frequent itemsets
            min_confidence: Minimum confidence threshold for association rules
            min_lift: Minimum lift threshold for association rules
        """
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_lift = min_lift
        self.rules = None
        self.frequent_itemsets = None
        self.transaction_data = None
        self.transactions = None
        
        # Optimization parameters
        self.max_transactions = 10000
        self.max_products = 200
        self.min_transaction_items = 2
        self.max_transaction_items = 50
    
    def prepare_transactions(self, df):
        """
        Prepare transaction data for market basket analysis with Phase 4.1 optimizations
        
        Args:
            df: DataFrame with columns ['InvoiceNo', 'Description']
        """
        print(f"📊 Starting MBA preparation with {len(df)} records...")
        
        # Phase 4.1 Optimization 1: Smart Sampling for large datasets
        optimized_df = self._optimize_dataset_size(df)
        
        # Phase 4.1 Optimization 2: Product filtering 
        optimized_df = self._filter_top_products(optimized_df)
        
        # Phase 4.1 Optimization 3: Transaction quality filtering
        optimized_df = self._filter_meaningful_transactions(optimized_df)
        
        # Group products by transaction
        transactions = optimized_df.groupby('InvoiceNo')['Description'].apply(list).tolist()
        
        # Convert to format required by mlxtend
        self.transactions = transactions
        
        # Create transaction encoder
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        self.transaction_df = pd.DataFrame(te_ary, columns=te.columns_)
        
        # Store the transaction data
        self.transaction_data = self.transaction_df
        
        print(f"✅ Optimized to {len(transactions)} transactions with {len(te.columns_)} unique products")
        print(f"📈 Data reduction: {len(df)} → {len(optimized_df)} records ({(1-len(optimized_df)/len(df))*100:.1f}% reduction)")
        print(f"🔢 Binary matrix shape: {self.transaction_data.shape}")
        
        return self.transaction_data
    
    def _optimize_dataset_size(self, df, max_transactions=None):
        """Phase 4.1 Optimization 1: Smart sampling for large datasets"""
        if max_transactions is None:
            max_transactions = self.max_transactions
            
        unique_transactions = df['InvoiceNo'].nunique()
        
        if unique_transactions > max_transactions:
            print(f"⚡ Large dataset detected ({unique_transactions} transactions)")
            print(f"   Sampling to {max_transactions} transactions for faster analysis...")
            
            # Sample transactions (not records) to maintain transaction integrity
            sample_transactions = df['InvoiceNo'].drop_duplicates().sample(
                n=max_transactions, 
                random_state=42
            )
            sampled_df = df[df['InvoiceNo'].isin(sample_transactions)]
            
            print(f"   ✅ Sampled {len(sampled_df)} records from {sample_transactions.nunique()} transactions")
            return sampled_df
        
        print(f"📊 Dataset size OK ({unique_transactions} transactions)")
        return df
    
    def _filter_top_products(self, df, max_products=None):
        """Phase 4.1 Optimization 2: Filter to only analyze most frequent products"""
        if max_products is None:
            max_products = self.max_products
            
        unique_products = df['Description'].nunique()
        
        if unique_products > max_products:
            print(f"🔍 Many products detected ({unique_products} unique products)")
            print(f"   Filtering to top {max_products} most frequent products...")
            
            # Get top products by frequency
            top_products = df['Description'].value_counts().head(max_products).index
            filtered_df = df[df['Description'].isin(top_products)]
            
            print(f"   ✅ Kept {len(filtered_df)} records with top {len(top_products)} products")
            return filtered_df
        
        print(f"📦 Product count OK ({unique_products} unique products)")
        return df
    
    def _filter_meaningful_transactions(self, df, min_items=None, max_items=None):
        """Phase 4.1 Optimization 3: Filter transactions by quality"""
        if min_items is None:
            min_items = self.min_transaction_items
        if max_items is None:
            max_items = self.max_transaction_items
            
        print(f"🧹 Filtering transaction quality (keeping {min_items}-{max_items} items per transaction)...")
        
        # Calculate transaction sizes
        transaction_sizes = df.groupby('InvoiceNo').size()
        
        # Filter transactions by size
        valid_transactions = transaction_sizes[
            (transaction_sizes >= min_items) & (transaction_sizes <= max_items)
        ].index
        
        filtered_df = df[df['InvoiceNo'].isin(valid_transactions)]
        
        removed_transactions = len(transaction_sizes) - len(valid_transactions)
        print(f"   ✅ Removed {removed_transactions} transactions (too small/large)")
        print(f"   ✅ Kept {len(valid_transactions)} quality transactions")
        
        return filtered_df
    
    def find_frequent_itemsets(self):
        """Find frequent itemsets using Apriori algorithm with progress indication"""
        if self.transaction_data is None:
            raise ValueError("Transaction data not prepared. Call prepare_transactions first.")
        
        print(f"🔍 Finding frequent itemsets (min_support={self.min_support})...")
        print(f"   Analyzing {self.transaction_data.shape[0]} transactions × {self.transaction_data.shape[1]} products")
        
        # Apply Apriori algorithm
        self.frequent_itemsets = apriori(
            self.transaction_data, 
            min_support=self.min_support, 
            use_colnames=True,
            verbose=1  # Show progress
        )
        
        if len(self.frequent_itemsets) == 0:
            print("⚠️ No frequent itemsets found. Try lowering min_support.")
            return self.frequent_itemsets
        
        print(f"✅ Found {len(self.frequent_itemsets)} frequent itemsets")
        return self.frequent_itemsets

    def generate_association_rules(self):
        """Generate association rules from frequent itemsets"""
        if self.frequent_itemsets is None or len(self.frequent_itemsets) == 0:
            print("❌ No frequent itemsets available for rule generation")
            return pd.DataFrame()
        
        print(f"📋 Generating association rules...")
        print(f"   Min confidence: {self.min_confidence}")
        print(f"   Min lift: {self.min_lift}")
        
        try:
            # Generate rules
            self.rules = association_rules(
                self.frequent_itemsets,
                metric="confidence",
                min_threshold=self.min_confidence
            )
            
            # Filter by lift
            self.rules = self.rules[self.rules['lift'] >= self.min_lift]
            
            if len(self.rules) == 0:
                print("⚠️ No rules found with current thresholds")
                return self.rules
            
            # Sort by lift descending
            self.rules = self.rules.sort_values('lift', ascending=False)
            
            # Convert frozensets to strings for better display
            self.rules['antecedents_str'] = self.rules['antecedents'].apply(
                lambda x: ', '.join(list(x))
            )
            self.rules['consequents_str'] = self.rules['consequents'].apply(
                lambda x: ', '.join(list(x))
            )
            
            print(f"✅ Generated {len(self.rules)} association rules")
            
            return self.rules
            
        except Exception as e:
            print(f"❌ Error generating rules: {str(e)}")
            return pd.DataFrame()

    def run_cached_analysis(self, df, data_hash=None):
        """
        Run complete MBA analysis with caching support
        
        Args:
            df: Transaction DataFrame
            data_hash: Optional hash of the data for cache key
            
        Returns:
            Tuple of (frequent_itemsets, association_rules, from_cache)
        """
        # Create cache parameters including optimization settings
        cache_params = {
            'min_support': self.min_support,
            'min_confidence': self.min_confidence,
            'min_lift': self.min_lift,
            'data_shape': df.shape,
            'data_hash': data_hash or str(hash(str(df.values.tobytes()))),
            'unique_products': df['Description'].nunique() if 'Description' in df.columns else 0,
            'max_transactions': self.max_transactions,
            'max_products': self.max_products,
            'min_transaction_items': self.min_transaction_items
        }
        
        # Try to load from cache
        cached_result = load_result("mba_results", cache_params)
        if cached_result is not None:
            print("✅ Loaded MBA results from cache")
            self.frequent_itemsets = cached_result['frequent_itemsets']
            self.rules = cached_result['rules']
            self.transaction_data = cached_result.get('transaction_data')
            return self.frequent_itemsets, self.rules, True
        
        print("⚠️ Computing MBA analysis... Please wait.")
        
        # Run the analysis with optimizations
        self.prepare_transactions(df)
        frequent_itemsets = self.find_frequent_itemsets()
        rules = self.generate_association_rules()
        
        # Save to cache
        result_to_cache = {
            'frequent_itemsets': frequent_itemsets,
            'rules': rules,
            'transaction_data': self.transaction_data
        }
        
        save_result("mba_results", cache_params, result_to_cache)
        
        # Clean up memory
        cleanup_memory()
        
        return frequent_itemsets, rules, False
    
    def get_product_recommendations(self, product_name, top_n=5):
        """
        Get cross-selling recommendations for a specific product
        
        Args:
            product_name: Name of the product to get recommendations for
            top_n: Number of top recommendations to return
        """
        if self.rules is None or len(self.rules) == 0:
            return []
        
        # Find rules where the product is in antecedents
        product_rules = self.rules[
            self.rules['antecedents'].apply(
                lambda x: product_name in x
            )
        ]
        
        if len(product_rules) == 0:
            return []
        
        # Get top recommendations
        recommendations = []
        for _, rule in product_rules.head(top_n).iterrows():
            rec = {
                'recommended_product': ', '.join(list(rule['consequents'])),
                'confidence': round(rule['confidence'], 3),
                'lift': round(rule['lift'], 3),
                'support': round(rule['support'], 3),
                'explanation': f"Customers who buy '{product_name}' also buy this {round(rule['confidence']*100, 1)}% of the time"
            }
            recommendations.append(rec)
        
        return recommendations
    
    def get_top_rules(self, top_n=10):
        """Get top association rules by lift"""
        if self.rules is None or len(self.rules) == 0:
            return []
        
        top_rules = []
        for _, rule in self.rules.head(top_n).iterrows():
            rule_info = {
                'antecedent': rule['antecedents_str'],
                'consequent': rule['consequents_str'],
                'confidence': round(rule['confidence'], 3),
                'lift': round(rule['lift'], 3),
                'support': round(rule['support'], 3),
                'rule': f"If customer buys {rule['antecedents_str']} → they will likely buy {rule['consequents_str']}"
            }
            top_rules.append(rule_info)
        
        return top_rules
    
    def get_analysis_summary(self):
        """Get summary statistics of the analysis"""
        if self.transaction_data is None:
            return {}
        
        summary = {
            'total_transactions': len(self.transaction_data),
            'unique_products': len(self.transaction_data.columns),
            'frequent_itemsets_count': len(self.frequent_itemsets) if self.frequent_itemsets is not None else 0,
            'association_rules_count': len(self.rules) if self.rules is not None else 0,
            'avg_basket_size': self.transaction_data.sum(axis=1).mean(),
            'min_support_used': self.min_support,
            'min_confidence_used': self.min_confidence,
            'min_lift_used': self.min_lift,
            'optimization_settings': {
                'max_transactions': self.max_transactions,
                'max_products': self.max_products,
                'min_transaction_items': self.min_transaction_items,
                'max_transaction_items': self.max_transaction_items
            }
        }
        
        return summary
    
    def update_optimization_settings(self, max_transactions=None, max_products=None, 
                                   min_transaction_items=None, max_transaction_items=None):
        """Update optimization parameters"""
        if max_transactions is not None:
            self.max_transactions = max_transactions
        if max_products is not None:
            self.max_products = max_products
        if min_transaction_items is not None:
            self.min_transaction_items = min_transaction_items
        if max_transaction_items is not None:
            self.max_transaction_items = max_transaction_items
        
        print("🔧 Updated optimization settings:")
        print(f"   Max transactions: {self.max_transactions}")
        print(f"   Max products: {self.max_products}")
        print(f"   Transaction size range: {self.min_transaction_items}-{self.max_transaction_items}")

if __name__ == "__main__":
    # Test optimized market basket analyzer
    print("Testing Optimized Market Basket Analyzer...")
    
    # Create sample data
    sample_data = pd.DataFrame({
        'InvoiceNo': ['001', '001', '002', '002', '003', '003', '004'],
        'Description': ['Product A', 'Product B', 'Product A', 'Product C', 
                       'Product B', 'Product C', 'Product A']
    })
    
    analyzer = MarketBasketAnalyzer(min_support=0.1)
    frequent_itemsets, rules, from_cache = analyzer.run_cached_analysis(sample_data)
    
    print("Optimized analysis completed!")
    print(f"From cache: {from_cache}")
    print(f"Frequent itemsets: {len(frequent_itemsets)}")
    print(f"Association rules: {len(rules)}")
