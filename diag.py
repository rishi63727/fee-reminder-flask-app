#!/usr/bin/env python3
"""
Diagnosis Script - Check what's wrong with your model predictions
Run this first to understand the issue
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

def diagnose_model():
    """Diagnose the model and feature mismatch issues"""
    
    print("🔍 DIAGNOSING FEE PREDICTION MODEL")
    print("=" * 50)
    
    # Check if model exists
    model_path = 'fee_prediction_model.pkl'
    if not os.path.exists(model_path):
        print("❌ Model file not found!")
        print("Please make sure 'fee_prediction_model.pkl' is in the same directory")
        return
    
    # Load model
    try:
        model = joblib.load(model_path)
        print(f"✅ Model loaded successfully: {type(model).__name__}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Check model features
    print("\n📊 MODEL FEATURE ANALYSIS")
    print("-" * 30)
    
    if hasattr(model, 'feature_names_in_'):
        expected_features = model.feature_names_in_
        print(f"Expected features: {len(expected_features)}")
        print("\nAll expected features:")
        for i, feature in enumerate(expected_features):
            print(f"  {i+1:2d}. {feature}")
    else:
        print("⚠️ No feature names stored in model")
        return
    
    # Show top important features if available
    if hasattr(model, 'feature_importances_'):
        print(f"\n🎯 TOP 10 MOST IMPORTANT FEATURES:")
        print("-" * 30)
        feature_importance = list(zip(expected_features, model.feature_importances_))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        for i, (feature, importance) in enumerate(feature_importance[:10]):
            print(f"  {i+1:2d}. {feature:<30} {importance:.4f}")
    
    # Create sample data to test predictions
    print(f"\n🧪 TESTING WITH SAMPLE DATA")
    print("-" * 30)
    
    # Create sample data that matches what your app provides
    sample_data = pd.DataFrame({
        'student_name': ['John Doe', 'Jane Smith', 'Bob Johnson'],
        'email': ['john@example.com', 'jane@example.com', 'bob@example.com'],
        'last_payment_date': ['2024-01-15', '2024-02-20', '2024-03-10'],
        'payment_plan': ['Annual', 'Semi-Annual', 'Annual'],
        'scholarship': ['Yes', 'No', 'Yes'],
        'past_late_payments': [0, 2, 1]
    })
    
    print("Sample input data:")
    print(sample_data)
    
    # Test current prediction method (like in your app)
    print(f"\n🔬 CURRENT PREDICTION METHOD RESULTS:")
    print("-" * 30)
    
    # Simulate your current app's feature preparation
    features_for_prediction = ['payment_plan', 'scholarship', 'past_late_payments']
    X_predict = sample_data[features_for_prediction].copy()
    
    # Convert past_late_payments to numeric
    X_predict['past_late_payments'] = pd.to_numeric(X_predict['past_late_payments'], errors='coerce').fillna(0)
    
    # One-hot encode
    X_predict = pd.get_dummies(X_predict, columns=['payment_plan', 'scholarship'], drop_first=True)
    
    print(f"Features after encoding: {list(X_predict.columns)}")
    print(f"Feature matrix shape: {X_predict.shape}")
    print("Feature matrix:")
    print(X_predict)
    
    # Check missing features
    available_features = set(X_predict.columns)
    required_features = set(expected_features)
    missing_features = required_features - available_features
    
    print(f"\n⚠️ FEATURE MISMATCH ANALYSIS:")
    print("-" * 30)
    print(f"Available features: {len(available_features)}")
    print(f"Required features: {len(required_features)}")
    print(f"Missing features: {len(missing_features)}")
    
    if missing_features:
        print(f"\nMissing features (first 10):")
        for i, feature in enumerate(list(missing_features)[:10]):
            print(f"  {i+1}. {feature}")
        
        if len(missing_features) > 10:
            print(f"  ... and {len(missing_features) - 10} more")
    
    # Add missing features with zeros (like your current app does)
    for feature in missing_features:
        X_predict[feature] = 0
    
    # Reorder to match model expectations
    X_predict_aligned = X_predict[expected_features]
    
    print(f"\nAfter adding missing features (all set to 0):")
    print(f"Final shape: {X_predict_aligned.shape}")
    
    # Make predictions
    try:
        probabilities = model.predict_proba(X_predict_aligned)[:, 1]
        
        print(f"\n📈 PREDICTION RESULTS:")
        print("-" * 30)
        for i, (name, prob) in enumerate(zip(sample_data['student_name'], probabilities)):
            risk = "High Risk" if prob >= 0.70 else "Medium Risk" if prob >= 0.30 else "Low Risk"
            print(f"  {name:<15} Probability: {prob:.4f} -> {risk}")
        
        print(f"\nProbability statistics:")
        print(f"  Mean: {probabilities.mean():.4f}")
        print(f"  Std:  {probabilities.std():.4f}")
        print(f"  Min:  {probabilities.min():.4f}")
        print(f"  Max:  {probabilities.max():.4f}")
        
        if probabilities.std() < 0.01:
            print("\n⚠️ WARNING: Very low variance - likely feature mismatch issue!")
        
        if probabilities.mean() > 0.8 or probabilities.mean() < 0.2:
            print("\n⚠️ WARNING: Extreme mean probability - model may be biased!")
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
    
    print(f"\n💡 RECOMMENDATIONS:")
    print("-" * 30)
    print("1. Your model expects {len(expected_features)} features but only gets {len(available_features)}")
    print("2. Missing features are filled with zeros, causing bias")
    print("3. You need to either:")
    print("   a) Retrain the model with fewer, more practical features, OR")
    print("   b) Enhance your prediction pipeline to create missing features")
    
    return model, expected_features, missing_features

if __name__ == "__main__":
    diagnose_model()