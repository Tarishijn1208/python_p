"""
Model Saver Script
==================
Add this code to the END of your project1.ipynb notebook to save all models
Copy and paste this entire script into a new cell at the bottom of your notebook
"""

import pickle
import os

print("=" * 60)
print("🔧 SAVING ALL MODELS FOR STREAMLIT APP")
print("=" * 60)

# Create a function to save models safely
def save_model(model, filename, model_name):
    """Save a model with error handling"""
    try:
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        print(f"✅ {model_name} saved successfully as {filename}")
        return True
    except Exception as e:
        print(f"❌ Error saving {model_name}: {str(e)}")
        return False

# Track success
saved_models = []

# 1. Save KNN Model
print("\n1. Saving KNN Model...")
if 'knn_model' in locals() or 'knn_model' in globals():
    if save_model(knn_model, 'knn_model.pkl', 'KNN Model'):
        saved_models.append('KNN')
else:
    print("⚠️  KNN model not found. Please train it first.")

# 2. Save Decision Tree Model
print("\n2. Saving Decision Tree Model...")
if 'dt_model' in locals() or 'dt_model' in globals():
    if save_model(dt_model, 'dt_model.pkl', 'Decision Tree Model'):
        saved_models.append('Decision Tree')
else:
    # Try alternative variable names
    alt_names = ['decision_tree_model', 'dt', 'tree_model']
    found = False
    for name in alt_names:
        if name in locals() or name in globals():
            model = locals().get(name) or globals().get(name)
            if save_model(model, 'dt_model.pkl', 'Decision Tree Model'):
                saved_models.append('Decision Tree')
            found = True
            break
    if not found:
        print("⚠️  Decision Tree model not found. Please train it first.")

# 3. Save Random Forest Model
print("\n3. Saving Random Forest Model...")
if 'rf_model' in locals() or 'rf_model' in globals():
    if save_model(rf_model, 'rf_model.pkl', 'Random Forest Model'):
        saved_models.append('Random Forest')
else:
    # Try alternative variable names
    alt_names = ['random_forest_model', 'rf', 'forest_model']
    found = False
    for name in alt_names:
        if name in locals() or name in globals():
            model = locals().get(name) or globals().get(name)
            if save_model(model, 'rf_model.pkl', 'Random Forest Model'):
                saved_models.append('Random Forest')
            found = True
            break
    if not found:
        print("⚠️  Random Forest model not found. Please train it first.")

# 4. Save Isolation Forest Model
print("\n4. Saving Isolation Forest Model...")
if 'iso_model' in locals() or 'iso_model' in globals():
    if save_model(iso_model, 'iso_model.pkl', 'Isolation Forest Model'):
        saved_models.append('Isolation Forest')
else:
    # Try alternative variable names
    alt_names = ['isolation_forest_model', 'iso', 'isolation_model']
    found = False
    for name in alt_names:
        if name in locals() or name in globals():
            model = locals().get(name) or globals().get(name)
            if save_model(model, 'iso_model.pkl', 'Isolation Forest Model'):
                saved_models.append('Isolation Forest')
            found = True
            break
    if not found:
        print("⚠️  Isolation Forest model not found. Please train it first.")

# 5. Save Logistic Regression Model
print("\n5. Saving Logistic Regression Model...")
if 'lr_model' in locals() or 'lr_model' in globals():
    if save_model(lr_model, 'lr_model.pkl', 'Logistic Regression Model'):
        saved_models.append('Logistic Regression')
else:
    # Try alternative variable names
    alt_names = ['logistic_regression_model', 'lr', 'log_reg_model', 'logistic_model']
    found = False
    for name in alt_names:
        if name in locals() or name in globals():
            model = locals().get(name) or globals().get(name)
            if save_model(model, 'lr_model.pkl', 'Logistic Regression Model'):
                saved_models.append('Logistic Regression')
            found = True
            break
    if not found:
        print("⚠️  Logistic Regression model not found. Please train it first.")

# 6. Save Feature Columns (CRITICAL!)
print("\n6. Saving Feature Columns...")
try:
    # Try to get columns from X_train
    if 'X_train' in locals() or 'X_train' in globals():
        X_train_obj = locals().get('X_train') or globals().get('X_train')
        model_columns = X_train_obj.columns.tolist()
        with open('model_columns.pkl', 'wb') as f:
            pickle.dump(model_columns, f)
        print(f"✅ Feature columns saved ({len(model_columns)} features)")
        print(f"   First 5 features: {model_columns[:5]}")
    else:
        print("⚠️  X_train not found. Trying alternative sources...")
        # Try X_train_smote or other variants
        if 'X_train_smote' in locals() or 'X_train_smote' in globals():
            X_obj = locals().get('X_train_smote') or globals().get('X_train_smote')
            model_columns = X_obj.columns.tolist()
            with open('model_columns.pkl', 'wb') as f:
                pickle.dump(model_columns, f)
            print(f"✅ Feature columns saved from X_train_smote ({len(model_columns)} features)")
        else:
            print("❌ Could not find feature columns. This is CRITICAL!")
            print("   Please save manually:")
            print("   >>> with open('model_columns.pkl', 'wb') as f:")
            print("   >>>     pickle.dump(X_train.columns.tolist(), f)")
except Exception as e:
    print(f"❌ Error saving feature columns: {str(e)}")

# Summary
print("\n" + "=" * 60)
print("📊 SUMMARY")
print("=" * 60)
print(f"✅ Successfully saved: {len(saved_models)}/5 models")
print(f"   Models: {', '.join(saved_models)}")
print()

# Check files exist
print("📁 Checking saved files:")
expected_files = ['knn_model.pkl', 'dt_model.pkl', 'rf_model.pkl', 
                  'iso_model.pkl', 'lr_model.pkl', 'model_columns.pkl']

existing_files = []
missing_files = []

for filename in expected_files:
    if os.path.exists(filename):
        file_size = os.path.getsize(filename) / 1024  # Size in KB
        print(f"   ✅ {filename} ({file_size:.1f} KB)")
        existing_files.append(filename)
    else:
        print(f"   ❌ {filename} - NOT FOUND")
        missing_files.append(filename)

print()
if len(existing_files) == len(expected_files):
    print("🎉 SUCCESS! All files saved. You're ready to run the Streamlit app!")
    print()
    print("Next steps:")
    print("1. Copy app.py, requirements.txt, and all .pkl files to the same folder")
    print("2. Copy final_dataset.xlsx to that folder")
    print("3. Run: streamlit run app.py")
else:
    print("⚠️  INCOMPLETE - Some files are missing:")
    for file in missing_files:
        print(f"   - {file}")
    print()
    print("Please train the missing models and run this script again.")

print("=" * 60)
