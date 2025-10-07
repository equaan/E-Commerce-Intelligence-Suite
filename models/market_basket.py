"""
Market Basket Analysis using Apriori Algorithm
Implements cross-selling recommendations for e-commerce products
"""

import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import warnings
warnings.filterwarnings('ignore')

class MarketBasketAnalyzer:
    def __init__(self, min_support=0.01, min_confidence=0.1, min_lift=1.0):
        """
        Initialize Market Basket Analyzer
        
        Args:
            min_support: Minimum support threshold for frequent itemsets
            min_confidence: Minimum confidence threshold for association rules
            min_lift: Minimum lift threshold for association rules
        """
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_lift = min_lift
        self.frequent_itemsets = None
        self.rules = None
        self.transaction_data = None
    
    def prepare_transactions(self, df):
        """
        Prepare transaction data for Apriori algorithm
        
        Args:
            df: DataFrame with columns ['InvoiceNo', 'Description']
        """
        print("🛒 Preparing transaction data for market basket analysis...")
        
        # Group products by invoice to create baskets
        transactions = df.groupby('InvoiceNo')['Description'].apply(list).tolist()
        
        # Filter transactions with at least 2 items
        transactions = [t for t in transactions if len(t) >= 2]
        
        print(f"   Created {len(transactions)} transaction baskets")
        
        # Use TransactionEncoder to create binary matrix
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        self.transaction_data = pd.DataFrame(te_ary, columns=te.columns_)
        
        print(f"   Binary matrix shape: {self.transaction_data.shape}")
        print(f"   Unique products: {len(te.columns_)}")
        
        return self.transaction_data
    
    def find_frequent_itemsets(self):
        """Find frequent itemsets using Apriori algorithm"""
        if self.transaction_data is None:
            raise ValueError("Transaction data not prepared. Call prepare_transactions first.")
        
        print(f"🔍 Finding frequent itemsets (min_support={self.min_support})...")
        
        # Apply Apriori algorithm
        self.frequent_itemsets = apriori(
            self.transaction_data, 
            min_support=self.min_support, 
            use_colnames=True
        )
        
        if len(self.frequent_itemsets) == 0:
            print("⚠️  No frequent itemsets found. Try lowering min_support.")
            return self.frequent_itemsets
        
        # Add length column
        self.frequent_itemsets['length'] = self.frequent_itemsets['itemsets'].apply(len)
        
        print(f"   Found {len(self.frequent_itemsets)} frequent itemsets")
        print(f"   Max itemset length: {self.frequent_itemsets['length'].max()}")
        
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
                print("⚠️  No rules found with current thresholds")
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
            'min_lift_used': self.min_lift
        }
        
        return summary
    
    def update_thresholds(self, min_support=None, min_confidence=None, min_lift=None):
        """Update analysis thresholds and regenerate rules"""
        if min_support is not None:
            self.min_support = min_support
        if min_confidence is not None:
            self.min_confidence = min_confidence
        if min_lift is not None:
            self.min_lift = min_lift
        
        print("🔄 Updating analysis with new thresholds...")
        
        # Regenerate analysis
        if self.transaction_data is not None:
            self.find_frequent_itemsets()
            self.generate_association_rules()

if __name__ == "__main__":
    # Test market basket analyzer
    print("Testing Market Basket Analyzer...")
    
    # Create sample data
    sample_data = pd.DataFrame({
        'InvoiceNo': ['001', '001', '002', '002', '003', '003', '004'],
        'Description': ['Product A', 'Product B', 'Product A', 'Product C', 
                       'Product B', 'Product C', 'Product A']
    })
    
    analyzer = MarketBasketAnalyzer(min_support=0.1)
    analyzer.prepare_transactions(sample_data)
    analyzer.find_frequent_itemsets()
    analyzer.generate_association_rules()
    
    print("Analysis completed!")
