#!/usr/bin/env python3
"""
Generate dummy historical fee payment data and save to Excel file
This creates realistic training data for the fee payment prediction model
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_training_data():
    """Generate comprehensive training data for fee payment prediction"""
    
    np.random.seed(42)
    random.seed(42)
    
    n_samples = 3000
    print(f"Generating {n_samples} student payment records...")
    
    data = []
    
    for i in range(n_samples):
        # Student Demographics
        student_id = f"STU{i+1:06d}"
        student_name = f"Student_{i+1}"
        email = f"student{i+1}@university.edu"
        
        # Academic Information
        program = random.choice(['Engineering', 'Business', 'Arts', 'Science', 'Medicine', 'Law'])
        semester = random.choice(['Fall 2023', 'Spring 2024', 'Fall 2024', 'Spring 2025'])
        academic_year = random.choice([2023, 2024, 2025])
        year_of_study = random.choice([1, 2, 3, 4, 5])  # 1=Freshman, 5=Graduate
        
        # GPA (affects payment behavior)
        gpa = round(np.random.normal(3.2, 0.7), 2)
        gpa = max(1.0, min(4.0, gpa))  # Clip to valid range
        
        # Financial Background
        family_income = int(np.random.lognormal(10.8, 0.6))  # Log-normal income distribution
        family_income = max(25000, min(150000, family_income))
        
        has_scholarship = random.choices([True, False], weights=[0.3, 0.7])[0]
        scholarship_amount = random.randint(1000, 5000) if has_scholarship else 0
        
        has_financial_aid = random.choices([True, False], weights=[0.4, 0.6])[0]
        financial_aid_amount = random.randint(2000, 8000) if has_financial_aid else 0
        
        # Employment status (affects ability to pay)
        is_employed = random.choices([True, False], weights=[0.35, 0.65])[0]
        monthly_income = random.randint(800, 2500) if is_employed else 0
        
        # Fee Structure
        base_fees = {
            'Engineering': 9500, 'Business': 8500, 'Medicine': 15000,
            'Law': 11000, 'Science': 7500, 'Arts': 6500
        }
        
        tuition_fee = base_fees[program] + random.randint(-500, 1000)
        library_fee = random.randint(200, 500)
        lab_fee = random.randint(300, 800) if program in ['Engineering', 'Science', 'Medicine'] else 0
        sports_fee = random.randint(100, 300)
        other_fees = random.randint(200, 600)
        
        total_fee_amount = tuition_fee + library_fee + lab_fee + sports_fee + other_fees
        net_amount_due = total_fee_amount - scholarship_amount - financial_aid_amount
        net_amount_due = max(1000, net_amount_due)  # Minimum amount
        
        # Payment Timeline
        semester_start = datetime(2024, 1, 15) if 'Spring' in semester else datetime(2023, 8, 15)
        due_date = semester_start + timedelta(days=random.randint(25, 45))
        
        # Calculate payment behavior based on risk factors
        risk_score = 0
        
        # Income risk (higher risk for lower income)
        if family_income < 40000: risk_score += 0.3
        elif family_income < 60000: risk_score += 0.15
        
        # Academic performance risk
        if gpa < 2.5: risk_score += 0.25
        elif gpa < 3.0: risk_score += 0.1
        
        # Employment risk
        if not is_employed and monthly_income == 0: risk_score += 0.2
        
        # Amount risk (higher amounts = higher risk)
        if net_amount_due > 8000: risk_score += 0.15
        elif net_amount_due > 6000: risk_score += 0.1
        
        # Year of study (seniors more likely to be late)
        if year_of_study >= 4: risk_score += 0.1
        
        # Financial aid dependency
        if not has_scholarship and not has_financial_aid: risk_score += 0.15
        
        # Generate actual payment behavior
        payment_probability = max(0.05, min(0.95, risk_score))
        will_pay_late = random.random() < payment_probability
        
        if will_pay_late:
            # Late payment - 1 to 45 days late
            days_late = int(np.random.exponential(8)) + 1  # Exponential distribution
            days_late = min(days_late, 60)  # Cap at 60 days
            actual_payment_date = due_date + timedelta(days=days_late)
            payment_status = "Late"
            is_late_payment = 1
        else:
            # On-time payment - pay 0-5 days early to 7 days late
            days_variation = random.randint(-5, 7)
            actual_payment_date = due_date + timedelta(days=days_variation)
            payment_status = "On Time" if days_variation <= 7 else "Late"
            is_late_payment = 1 if days_variation > 7 else 0
        
        # Payment method
        payment_method = random.choices(
            ['Online', 'Bank Transfer', 'Cash', 'Check', 'Credit Card'],
            weights=[0.4, 0.25, 0.15, 0.1, 0.1]
        )[0]
        
        # Previous payment history (affects current behavior)
        previous_late_payments = random.choices([0, 1, 2, 3, 4], weights=[0.5, 0.25, 0.15, 0.07, 0.03])[0]
        has_payment_plan = random.choices([True, False], weights=[0.2, 0.8])[0]
        
        # Contact information completeness
        phone_provided = random.choices([True, False], weights=[0.85, 0.15])[0]
        address_complete = random.choices([True, False], weights=[0.9, 0.1])[0]
        emergency_contact = random.choices([True, False], weights=[0.7, 0.3])[0]
        
        # Distance from university (affects payment behavior)
        distance_from_campus = random.choices(
            ['Local', 'Regional', 'National', 'International'],
            weights=[0.4, 0.3, 0.25, 0.05]
        )[0]
        
        # Create record
        record = {
            'student_id': student_id,
            'student_name': student_name,
            'email': email,
            'program': program,
            'semester': semester,
            'academic_year': academic_year,
            'year_of_study': year_of_study,
            'gpa': gpa,
            'family_income': family_income,
            'has_scholarship': has_scholarship,
            'scholarship_amount': scholarship_amount,
            'has_financial_aid': has_financial_aid,
            'financial_aid_amount': financial_aid_amount,
            'is_employed': is_employed,
            'monthly_income': monthly_income,
            'tuition_fee': tuition_fee,
            'library_fee': library_fee,
            'lab_fee': lab_fee,
            'sports_fee': sports_fee,
            'other_fees': other_fees,
            'total_fee_amount': total_fee_amount,
            'net_amount_due': net_amount_due,
            'due_date': due_date.strftime('%Y-%m-%d'),
            'actual_payment_date': actual_payment_date.strftime('%Y-%m-%d'),
            'payment_status': payment_status,
            'is_late_payment': is_late_payment,  # TARGET VARIABLE
            'payment_method': payment_method,
            'previous_late_payments': previous_late_payments,
            'has_payment_plan': has_payment_plan,
            'phone_provided': phone_provided,
            'address_complete': address_complete,
            'emergency_contact': emergency_contact,
            'distance_from_campus': distance_from_campus,
            'days_late': (actual_payment_date - due_date).days
        }
        
        data.append(record)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add some additional calculated features
    df['amount_per_gpa'] = df['net_amount_due'] / df['gpa']
    df['aid_coverage_ratio'] = (df['scholarship_amount'] + df['financial_aid_amount']) / df['total_fee_amount']
    df['income_to_fee_ratio'] = df['family_income'] / df['net_amount_due']
    
    print(f"Dataset created with {len(df)} records")
    print(f"Late payment rate: {df['is_late_payment'].mean():.2%}")
    print(f"Columns: {len(df.columns)}")
    
    return df

def save_to_excel():
    """Generate data and save to Excel file"""
    
    df = generate_training_data()
    
    # Save to Excel
    filename = 'fee_payment_training_data.xlsx'
    df.to_excel(filename, index=False, sheet_name='Payment_History')
    
    print(f"\n✅ Data saved to '{filename}'")
    print("\nDataset Summary:")
    print(f"- Total Records: {len(df)}")
    print(f"- Features: {len(df.columns)}")
    print(f"- Late Payment Rate: {df['is_late_payment'].mean():.1%}")
    print(f"- Date Range: {df['due_date'].min()} to {df['due_date'].max()}")
    
    print(f"\nTarget Variable: 'is_late_payment'")
    print(f"- 0 = On-time payment")
    print(f"- 1 = Late payment (more than 7 days)")
    
    print(f"\nKey Features Include:")
    print(f"- Student demographics (GPA, program, year)")
    print(f"- Financial information (income, aid, fees)")
    print(f"- Payment history (previous late payments)")
    print(f"- Contact completeness indicators")
    
    return filename

if __name__ == "__main__":
    filename = save_to_excel()
    print(f"\n🎯 Ready to train! Use this file: {filename}")
    print(f"Run the training script and provide this Excel file path when prompted.")