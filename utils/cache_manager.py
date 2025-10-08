"""
Persistent Disk-Based Caching System for E-Commerce Intelligence Suite
Handles caching of MBA and Time-Series model results to reduce memory usage
"""

import os
import pickle
import hashlib
import json
import gzip
import gc
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

class CacheManager:
    def __init__(self, base_cache_dir="cache_logs"):
        """Initialize cache manager with base directory"""
        self.base_cache_dir = base_cache_dir
        self.log_file = os.path.join(base_cache_dir, "log.txt")
        
        # Create cache directories
        self.mba_cache_dir = os.path.join(base_cache_dir, "mba_results")
        self.arima_cache_dir = os.path.join(base_cache_dir, "arima_results")
        
        # Ensure directories exist
        os.makedirs(self.mba_cache_dir, exist_ok=True)
        os.makedirs(self.arima_cache_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
    
    def get_hash(self, params_dict: Dict[str, Any]) -> str:
        """
        Generate unique MD5 hash from parameter dictionary
        
        Args:
            params_dict: Dictionary of parameters to hash
            
        Returns:
            MD5 hash string
        """
        # Convert to JSON string with sorted keys for consistent hashing
        params_json = json.dumps(params_dict, sort_keys=True, default=str)
        return hashlib.md5(params_json.encode()).hexdigest()
    
    def _get_cache_path(self, algorithm: str, params_hash: str, compressed: bool = True) -> str:
        """Get the full path for cache file"""
        if algorithm == "mba_results":
            cache_dir = self.mba_cache_dir
        elif algorithm == "arima_results":
            cache_dir = self.arima_cache_dir
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        extension = ".pkl.gz" if compressed else ".pkl"
        return os.path.join(cache_dir, f"{params_hash}{extension}")
    
    def save_result(self, algorithm: str, params: Dict[str, Any], result: Any, compress: bool = True) -> bool:
        """
        Save model result to disk cache
        
        Args:
            algorithm: Algorithm type ('mba_results' or 'arima_results')
            params: Parameter dictionary used for the computation
            result: Result object to cache
            compress: Whether to compress the file with gzip
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            params_hash = self.get_hash(params)
            cache_path = self._get_cache_path(algorithm, params_hash, compress)
            
            # Save with compression if requested
            if compress:
                with gzip.open(cache_path, 'wb') as f:
                    pickle.dump(result, f)
            else:
                with open(cache_path, 'wb') as f:
                    pickle.dump(result, f)
            
            # Log the cache save
            self._log_cache_operation("SAVE", algorithm, params_hash, params)
            
            # Force garbage collection to free memory
            del result
            gc.collect()
            
            return True
            
        except Exception as e:
            print(f"Error saving cache: {str(e)}")
            return False
    
    def load_result(self, algorithm: str, params: Dict[str, Any]) -> Optional[Any]:
        """
        Load model result from disk cache
        
        Args:
            algorithm: Algorithm type ('mba_results' or 'arima_results')
            params: Parameter dictionary to look for
            
        Returns:
            Cached result if found, None otherwise
        """
        try:
            params_hash = self.get_hash(params)
            
            # Try compressed file first
            cache_path_gz = self._get_cache_path(algorithm, params_hash, compressed=True)
            cache_path = self._get_cache_path(algorithm, params_hash, compressed=False)
            
            result = None
            
            if os.path.exists(cache_path_gz):
                with gzip.open(cache_path_gz, 'rb') as f:
                    result = pickle.load(f)
                self._log_cache_operation("HIT", algorithm, params_hash, params)
                
            elif os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    result = pickle.load(f)
                self._log_cache_operation("HIT", algorithm, params_hash, params)
                
            else:
                self._log_cache_operation("MISS", algorithm, params_hash, params)
                return None
            
            return result
            
        except Exception as e:
            print(f"Error loading cache: {str(e)}")
            self._log_cache_operation("ERROR", algorithm, "unknown", params)
            return None
    
    def _log_cache_operation(self, operation: str, algorithm: str, params_hash: str, params: Dict[str, Any]):
        """Log cache operations for debugging and monitoring"""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"{timestamp} | {operation} | {algorithm} | {params_hash} | {json.dumps(params, default=str)}\n"
            
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_entry)
                
        except Exception as e:
            print(f"Error logging cache operation: {str(e)}")
    
    def clear_cache(self, algorithm: Optional[str] = None) -> bool:
        """
        Clear cache files
        
        Args:
            algorithm: Specific algorithm to clear, or None for all
            
        Returns:
            True if cleared successfully
        """
        try:
            if algorithm is None:
                # Clear all caches
                for cache_dir in [self.mba_cache_dir, self.arima_cache_dir]:
                    for file in os.listdir(cache_dir):
                        if file.endswith(('.pkl', '.pkl.gz')):
                            os.remove(os.path.join(cache_dir, file))
            else:
                # Clear specific algorithm cache
                if algorithm == "mba_results":
                    cache_dir = self.mba_cache_dir
                elif algorithm == "arima_results":
                    cache_dir = self.arima_cache_dir
                else:
                    return False
                
                for file in os.listdir(cache_dir):
                    if file.endswith(('.pkl', '.pkl.gz')):
                        os.remove(os.path.join(cache_dir, file))
            
            # Log the clear operation
            self._log_cache_operation("CLEAR", algorithm or "ALL", "N/A", {})
            
            # Force garbage collection
            gc.collect()
            
            return True
            
        except Exception as e:
            print(f"Error clearing cache: {str(e)}")
            return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            stats = {
                "mba_cache_files": len([f for f in os.listdir(self.mba_cache_dir) if f.endswith(('.pkl', '.pkl.gz'))]),
                "arima_cache_files": len([f for f in os.listdir(self.arima_cache_dir) if f.endswith(('.pkl', '.pkl.gz'))]),
                "total_cache_size_mb": 0
            }
            
            # Calculate total cache size
            for cache_dir in [self.mba_cache_dir, self.arima_cache_dir]:
                for file in os.listdir(cache_dir):
                    if file.endswith(('.pkl', '.pkl.gz')):
                        file_path = os.path.join(cache_dir, file)
                        stats["total_cache_size_mb"] += os.path.getsize(file_path)
            
            stats["total_cache_size_mb"] = round(stats["total_cache_size_mb"] / (1024 * 1024), 2)
            
            return stats
            
        except Exception as e:
            print(f"Error getting cache stats: {str(e)}")
            return {"error": str(e)}

# Global cache manager instance
cache_manager = CacheManager()

# Convenience functions for easy import
def get_hash(params_dict: Dict[str, Any]) -> str:
    """Generate unique hash from parameter dictionary"""
    return cache_manager.get_hash(params_dict)

def save_result(algorithm: str, params: Dict[str, Any], result: Any) -> bool:
    """Save model result to disk cache"""
    return cache_manager.save_result(algorithm, params, result)

def load_result(algorithm: str, params: Dict[str, Any]) -> Optional[Any]:
    """Load model result from disk cache"""
    return cache_manager.load_result(algorithm, params)

def clear_cache(algorithm: Optional[str] = None) -> bool:
    """Clear cache files"""
    return cache_manager.clear_cache(algorithm)

def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics"""
    return cache_manager.get_cache_stats()

def cleanup_memory():
    """Force garbage collection to free memory"""
    gc.collect()
