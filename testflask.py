#!/usr/bin/env python3
"""
Test script to verify the Flask app setup
"""

import os
import pandas as pd
import joblib
from datetime import datetime

def test_model_loading():
    """Test if the model can be loaded"""
    print("🧪 Testing Model Loading...")
    try:
        model = joblib.load('fee_prediction_model.pkl')
        print("✅ Model loaded successfully!")
        print(f"   Model type: {type(model)}")
        if hasattr(model, 'feature_names_in_'):
            print(f"   Expected features: {list(model.feature_names_in_)}")
        return True
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def test_file_structure():
    """Test if required files and folders exist"""
    print("\n🧪 Testing File Structure...")
    
    required_files = [
        'fee_prediction_model.pkl',
        'templates/index.html',
        'templates/mail_log.html',
        'static/css/styles.css',
        'static/js/script.js'
    ]
    
    all_good = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} - Found")
        else:
            print(f"❌ {file_path} - Missing")
            all_good = False
    
    # Check if uploads directory can be created
    try:
        os.makedirs('uploads', exist_ok=True)
        print("✅ uploads/ - Directory ready")
    except Exception as e:
        print(f"❌ uploads/ - Cannot create: {e}")
        all_good = False
    
    return all_good

def create_sample_data():
    """Create a sample Excel file for testing"""
    print("\n🧪 Creating Sample Test Data...")
    
    sample_data = {
        'student_name': ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice Brown'],
        'email': ['john@example.com', 'jane@example.com', 'bob@example.com', 'alice@example.com'],
        'last_payment_date': ['2024-01-15', '2024-02-10', '2023-12-01', '2024-03-05'],
        'payment_plan': ['Annual', 'Semi-Annual', 'Annual', 'Semi-Annual'],
        'scholarship': ['Yes', 'No', 'No', 'Yes'],
        'past_late_payments': [0, 2, 1, 0]
    }
    
    df = pd.DataFrame(sample_data)
    df['last_payment_date'] = pd.to_datetime(df['last_payment_date'])
    
    try:
        df.to_excel('sample_student_data.xlsx', index=False)
        print("✅ Created sample_student_data.xlsx for testing")
        print("   Columns:", list(df.columns))
        print(f"   Rows: {len(df)}")
        return True
    except Exception as e:
        print(f"❌ Failed to create sample data: {e}")
        return False

def test_data_processing():
    """Test data processing functionality"""
    print("\n🧪 Testing Data Processing...")
    
    if not os.path.exists('sample_student_data.xlsx'):
        print("❌ Sample data file not found")
        return False
    
    try:
        # Load and process like the app would
        df = pd.read_excel('sample_student_data.xlsx', engine="openpyxl")
        
        # Normalize columns
        df.columns = [col.strip().replace(' ', '_').lower() for col in df.columns]
        
        # Check required columns
        required_columns = ['student_name', 'email', 'last_payment_date', 'payment_plan', 'scholarship', 'past_late_payments']
        missing = [col for col in required_columns if col not in df.columns]
        
        if missing:
            print(f"❌ Missing columns: {missing}")
            return False
        
        print("✅ Data processing test passed")
        print(f"   Processed {len(df)} records")
        return True
        
    except Exception as e:
        print(f"❌ Data processing test failed: {e}")
        return False

def test_flask_imports():
    """Test if all Flask dependencies can be imported"""
    print("\n🧪 Testing Flask Dependencies...")
    
    try:
        import flask
        import pandas
        import joblib
        from dateutil.relativedelta import relativedelta
        from flask_mail import Mail
        print("✅ All required packages can be imported")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Run: pip install flask pandas joblib python-dateutil flask-mail openpyxl")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("🚀 FLASK APP SETUP TESTING")
    print("=" * 60)
    
    tests = [
        ("Flask Dependencies", test_flask_imports),
        ("Model Loading", test_model_loading),
        ("File Structure", test_file_structure),
        ("Sample Data Creation", create_sample_data),
        ("Data Processing", test_data_processing)
    ]
    
    results = {}
    for test_name, test_func in tests:
        results[test_name] = test_func()
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:.<30} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("\n🎉 All tests passed! Your Flask app should work.")
        print("\nNext steps:")
        print("1. Run: python app.py")
        print("2. Open: http://localhost:5000")
        print("3. Upload: sample_student_data.xlsx")
    else:
        print("\n⚠️ Some tests failed. Please fix the issues above.")

if __name__ == "__main__":
    main()