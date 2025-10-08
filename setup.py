"""
Setup script for E-Commerce Intelligence Suite
Helps users get started quickly with their own data
"""

import os
import sys
import subprocess
from pathlib import Path

def print_header():
    """Print setup header"""
    print("🚀 E-Commerce Intelligence Suite - Setup Assistant")
    print("=" * 60)
    print("This script will help you set up the project with your data")
    print("=" * 60)

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} detected")
        print("   This project requires Python 3.8 or higher")
        print("   Please upgrade Python and try again")
        return False
    else:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible!")
        return True

def install_dependencies():
    """Install required Python packages"""
    print("\n📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        print("   Please run manually: pip install -r requirements.txt")
        return False

def setup_data_directory():
    """Create data directory if it doesn't exist"""
    print("\n📁 Setting up data directory...")
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    print("✅ Data directory ready!")

def configure_csv_path():
    """Help user configure their CSV file path"""
    print("\n📊 CSV Data Configuration")
    print("-" * 30)
    
    # Check if sample data exists
    sample_path = "data/sample_retail_data.csv"
    if os.path.exists(sample_path):
        print(f"✅ Sample data found at: {sample_path}")
        use_sample = input("Do you want to use the sample data for now? (y/n): ").lower().strip()
        if use_sample in ['y', 'yes']:
            return sample_path
    
    print("\n📝 Please provide the path to your CSV file:")
    print("Examples:")
    print("  - data/my_sales_data.csv")
    print("  - C:/Users/YourName/Downloads/sales.csv")
    print("  - /path/to/your/ecommerce_data.csv")
    
    while True:
        csv_path = input("\nEnter CSV file path: ").strip()
        if not csv_path:
            print("❌ Please enter a valid path")
            continue
        
        if os.path.exists(csv_path):
            print(f"✅ CSV file found: {csv_path}")
            return csv_path
        else:
            print(f"❌ File not found: {csv_path}")
            retry = input("Try again? (y/n): ").lower().strip()
            if retry not in ['y', 'yes']:
                return None

def update_config_file(csv_path):
    """Update config.py with user's CSV path"""
    print(f"\n⚙️ Updating configuration with CSV path: {csv_path}")
    
    try:
        # Read current config (UTF-8 safe for emojis)
        with open('config.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Update CSV_FILE_PATH line
        lines = content.split('\n')
        updated = False
        for i, line in enumerate(lines):
            if line.strip().startswith('CSV_FILE_PATH = '):
                lines[i] = f'CSV_FILE_PATH = "{csv_path}"'
                updated = True
                break
        
        if not updated:
            # If the variable isn't found, append it at the top
            lines.insert(0, f'CSV_FILE_PATH = "{csv_path}"')

        # Write updated config (UTF-8 again)
        with open('config.py', 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        print("✅ Configuration updated successfully!")
        return True

    except Exception as e:
        print(f"❌ Error updating configuration: {str(e)}")
        return False

def run_initialization():
    """Run the data initialization script"""
    print("\n🗄️ Initializing database with your data...")
    print("This may take a few minutes depending on your data size...")
    
    try:
        result = subprocess.run([sys.executable, "initialize_data.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Database initialization completed!")
            print("\nOutput:")
            print(result.stdout)
            return True
        else:
            print("❌ Database initialization failed!")
            print("\nError:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error running initialization: {str(e)}")
        return False

def main():
    """Main setup function"""
    print_header()
    
    # Step 1: Check Python version
    if not check_python_version():
        return False
    
    # Step 2: Install dependencies
    if not install_dependencies():
        return False
    
    # Step 3: Setup data directory
    setup_data_directory()
    
    # Step 4: Configure CSV path
    csv_path = configure_csv_path()
    if not csv_path:
        print("❌ Setup cancelled - no CSV file configured")
        return False
    
    # Step 5: Update config file
    if not update_config_file(csv_path):
        return False
    
    # Step 6: Initialize database
    if not run_initialization():
        print("\n🔧 Manual steps to complete setup:")
        print("   1. Check your CSV file format matches requirements")
        print("   2. Update column mappings in config.py if needed")
        print("   3. Run: python initialize_data.py")
        return False
    
    # Success!
    print("\n" + "=" * 60)
    print("🎉 SETUP COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("🚀 Next steps:")
    print("   1. Run: streamlit run app.py")
    print("   2. Open your browser to the provided URL")
    print("   3. Start exploring your e-commerce data!")
    print("=" * 60)
    
    # Ask if user wants to start the app now
    start_now = input("\nWould you like to start the application now? (y/n): ").lower().strip()
    if start_now in ['y', 'yes']:
        print("\n🚀 Starting Streamlit application...")
        try:
            subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
        except KeyboardInterrupt:
            print("\n👋 Application stopped by user")
        except Exception as e:
            print(f"\n❌ Error starting application: {str(e)}")
            print("   Please run manually: streamlit run app.py")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            print("\n❌ Setup failed. Please check the errors above and try again.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n👋 Setup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error during setup: {str(e)}")
        sys.exit(1)
