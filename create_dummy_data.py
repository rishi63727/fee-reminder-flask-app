#!/usr/bin/env python3
"""
Generate realistic dummy student fee data for testing the Flask Fee Reminder System
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from faker import Faker
import os

# Initialize Faker for generating realistic names and emails
fake = Faker()

def generate_student_data(num_students=100):
    """Generate realistic student fee data"""
    
    print(f"🏗️  Generating {num_students} student records...")
    
    # Set random seed for reproducible results
    random.seed(42)
    np.random.seed(42)
    Faker.seed(42)
    
    data = {
        'student_name': [],
        'email': [],
        'last_payment_date': [],
        'payment_plan': [],
        'scholarship': [],
        'past_late_payments': [],
        'student_id': [],
        'program': [],
        'year_of_study': [],
        'fee_amount': []
    }
    
    # Payment plans with realistic distribution
    payment_plans = ['Annual', 'Semi-Annual']
    payment_plan_weights = [0.7, 0.3]  # 70% Annual, 30% Semi-Annual
    
    # Scholarship distribution
    scholarship_options = ['Yes', 'No']
    scholarship_weights = [0.25, 0.75]  # 25% have scholarships
    
    # Academic programs
    programs = [
        'Computer Science', 'Business Administration', 'Engineering', 
        'Medicine', 'Law', 'Arts', 'Psychology', 'Mathematics',
        'Physics', 'Chemistry', 'Biology', 'Economics'
    ]
    
    # Generate data for each student
    for i in range(num_students):
        # Basic info
        name = fake.name()
        # Create email from name
        email_name = name.lower().replace(' ', '.').replace("'", "")
        email = f"{email_name}@university.edu"
        
        data['student_name'].append(name)
        data['email'].append(email)
        data['student_id'].append(f"STU{2024}{str(i+1).zfill(4)}")
        
        # Academic info
        data['program'].append(random.choice(programs))
        data['year_of_study'].append(random.randint(1, 4))
        
        # Payment plan
        payment_plan = np.random.choice(payment_plans, p=payment_plan_weights)
        data['payment_plan'].append(payment_plan)
        
        # Scholarship
        scholarship = np.random.choice(scholarship_options, p=scholarship_weights)
        data['scholarship'].append(scholarship)
        
        # Past late payments (0-5, with most students having 0-2)
        past_late = np.random.choice([0, 1, 2, 3, 4, 5], p=[0.4, 0.3, 0.15, 0.1, 0.04, 0.01])
        data['past_late_payments'].append(past_late)
        
        # Fee amount based on program and scholarship
        base_fee = random.randint(8000, 15000)  # Base fee range
        if scholarship == 'Yes':
            fee_amount = base_fee * 0.5  # 50% discount for scholarship
        else:
            fee_amount = base_fee
        data['fee_amount'].append(round(fee_amount, 2))
        
        # Last payment date - realistic distribution
        # Most recent payments within last 6-18 months
        days_ago = random.randint(30, 500)  # 1 month to ~16 months ago
        last_payment = datetime.now() - timedelta(days=days_ago)
        data['last_payment_date'].append(last_payment.strftime('%Y-%m-%d'))
    
    return pd.DataFrame(data)

def create_test_scenarios():
    """Create specific test scenarios for different use cases"""
    
    print("🎯 Creating specific test scenarios...")
    
    # Scenario 1: Mixed risk levels
    scenario1_data = {
        'student_name': [
            'Alice Johnson',      # Low risk - recent payment, no late history
            'Bob Smith',          # Medium risk - moderate late history
            'Carol Davis',        # High risk - many late payments, old payment
            'David Wilson',       # Medium risk - scholarship but some late payments
            'Emma Brown'          # Low risk - scholarship, good history
        ],
        'email': [
            'alice.johnson@university.edu',
            'bob.smith@university.edu', 
            'carol.davis@university.edu',
            'david.wilson@university.edu',
            'emma.brown@university.edu'
        ],
        'student_id': ['TEST001', 'TEST002', 'TEST003', 'TEST004', 'TEST005'],
        'last_payment_date': [
            '2024-06-01',  # Recent payment
            '2024-03-15',  # Moderate
            '2023-12-01',  # Old payment
            '2024-04-20',  # Moderate
            '2024-07-10'   # Very recent
        ],
        'payment_plan': ['Annual', 'Semi-Annual', 'Annual', 'Semi-Annual', 'Annual'],
        'scholarship': ['No', 'No', 'No', 'Yes', 'Yes'],
        'past_late_payments': [0, 2, 5, 1, 0],
        'program': ['Computer Science', 'Business', 'Engineering', 'Medicine', 'Arts'],
        'year_of_study': [2, 3, 4, 1, 2],
        'fee_amount': [12000, 10000, 15000, 7500, 6000]
    }
    
    return pd.DataFrame(scenario1_data)

def generate_multiple_datasets():
    """Generate multiple datasets for different testing scenarios"""
    
    datasets = {}
    
    # Dataset 1: Small test set (5 students) - for quick testing
    print("\n📋 Creating small test dataset...")
    datasets['small_test'] = create_test_scenarios()
    
    # Dataset 2: Medium set (50 students) - for demo
    print("📋 Creating medium demo dataset...")
    datasets['demo'] = generate_student_data(50)
    
    # Dataset 3: Large set (200 students) - for performance testing  
    print("📋 Creating large test dataset...")
    datasets['large'] = generate_student_data(200)
    
    # Dataset 4: Edge cases
    print("📋 Creating edge cases dataset...")
    edge_cases = generate_edge_cases()
    datasets['edge_cases'] = edge_cases
    
    return datasets

def generate_edge_cases():
    """Generate edge cases for thorough testing"""
    
    edge_data = {
        'student_name': [
            "O'Connor, Mary-Jane",      # Special characters in name
            "José García-López",        # International characters
            "李小明",                    # Non-Latin characters
            "Very Long Student Name That Might Cause Issues",  # Long name
            "Student WithoutEmail",     # Will need email handling
        ],
        'email': [
            "mary.oconnor@university.edu",
            "jose.garcia@university.edu", 
            "xm.li@university.edu",
            "very.long.name@university.edu",
            "no.email@university.edu"
        ],
        'student_id': ['EDGE001', 'EDGE002', 'EDGE003', 'EDGE004', 'EDGE005'],
        'last_payment_date': [
            '2020-01-01',  # Very old payment
            '2024-12-01',  # Future payment (edge case)
            '2024-08-01',  # Recent payment
            '',            # Empty date (will test error handling)
            '2023-06-15'   # Moderate
        ],
        'payment_plan': ['Annual', 'Semi-Annual', 'Annual', 'Semi-Annual', 'Annual'],
        'scholarship': ['No', 'Yes', 'No', 'Yes', 'No'],
        'past_late_payments': [10, 0, 3, 999, 1],  # Including extreme values
        'program': ['Computer Science', 'Engineering', 'Medicine', 'Business', 'Arts'],
        'year_of_study': [1, 2, 3, 4, 1],
        'fee_amount': [15000, 7500, 12000, 0, 10000]  # Including zero fee
    }
    
    return pd.DataFrame(edge_data)

def save_datasets(datasets):
    """Save all datasets to Excel files"""
    
    print("\n💾 Saving datasets to Excel files...")
    
    # Create data directory
    os.makedirs('sample_data', exist_ok=True)
    
    for name, df in datasets.items():
        filename = f'sample_data/student_data_{name}.xlsx'
        
        try:
            # Save with additional formatting
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Student_Data', index=False)
                
                # Add a summary sheet
                summary_df = create_dataset_summary(df)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            print(f"✅ Saved {filename} ({len(df)} records)")
            
        except Exception as e:
            print(f"❌ Error saving {filename}: {e}")

def create_dataset_summary(df):
    """Create a summary of the dataset"""
    
    summary_data = {
        'Metric': [
            'Total Students',
            'Students with Scholarships',
            'Annual Payment Plans', 
            'Semi-Annual Payment Plans',
            'Students with 0 Late Payments',
            'Students with 1+ Late Payments',
            'Average Fee Amount',
            'Date Range (Last Payment)'
        ],
        'Value': [
            len(df),
            len(df[df['scholarship'] == 'Yes']),
            len(df[df['payment_plan'] == 'Annual']),
            len(df[df['payment_plan'] == 'Semi-Annual']),
            len(df[df['past_late_payments'] == 0]),
            len(df[df['past_late_payments'] > 0]),
            f"${df['fee_amount'].mean():.2f}",
            f"{df['last_payment_date'].min()} to {df['last_payment_date'].max()}"
        ]
    }
    
    return pd.DataFrame(summary_data)

def main():
    """Main function to generate all dummy data"""
    
    print("=" * 60)
    print("🎓 STUDENT FEE DATA GENERATOR")
    print("=" * 60)
    
    # Check if faker is available
    try:
        from faker import Faker
    except ImportError:
        print("❌ Faker library not found. Installing...")
        print("Run: pip install faker")
        return
    
    # Generate datasets
    datasets = generate_multiple_datasets()
    
    # Save to files
    save_datasets(datasets)
    
    print("\n" + "=" * 60)
    print("📊 GENERATION COMPLETE!")
    print("=" * 60)
    
    print("\nFiles created in 'sample_data/' directory:")
    print("• student_data_small_test.xlsx    - 5 students (quick testing)")
    print("• student_data_demo.xlsx          - 50 students (demo/presentation)")
    print("• student_data_large.xlsx         - 200 students (performance testing)")
    print("• student_data_edge_cases.xlsx    - 5 students (edge case testing)")
    
    print("\nColumns included:")
    print("• student_name          - Full name")
    print("• email                 - University email")
    print("• student_id            - Unique ID")
    print("• last_payment_date     - Date of last payment")
    print("• payment_plan          - Annual/Semi-Annual")
    print("• scholarship           - Yes/No")
    print("• past_late_payments    - Number of previous late payments")
    print("• program               - Academic program")
    print("• year_of_study         - 1-4")
    print("• fee_amount            - Fee amount in dollars")
    
    print("\n🚀 Ready to test your Flask app!")
    print("1. Start your Flask app: python app.py")
    print("2. Upload any of the generated Excel files")
    print("3. View the risk analysis results")

if __name__ == "__main__":
    main()