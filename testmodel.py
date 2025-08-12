import joblib
import os

# Check if file exists
model_path = 'fee_prediction_model.pkl'
print(f"Model file exists: {os.path.exists(model_path)}")

# Try loading
try:
    model = joblib.load(model_path)
    print("✅ Model loaded successfully!")
    print(f"Model type: {type(model)}")
    if hasattr(model, 'feature_names_in_'):
        print(f"Expected features: {model.feature_names_in_}")
except Exception as e:
    print(f"❌ Model loading failed: {e}")