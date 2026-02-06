
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

print("🚀 Starting model training...")

# 1. Load Data
print("📂 Loading dataset...")
try:
    df = pd.read_excel('data/final_dataset.xlsx')
    print(f"   Data loaded: {df.shape}")
except Exception as e:
    print(f"❌ Error loading data: {e}")
    exit()

# 2. Preprocessing
print("🧹 Preprocessing data...")
df.drop_duplicates(inplace=True)

# Drop ID columns
id_cols = ['patient_id', 'claim_id', 'policy_number']
df.drop(columns=[col for col in id_cols if col in df.columns], inplace=True)

# Handle missing values
# Numeric columns: median
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# Categorical columns: mode
object_cols = df.select_dtypes(include=['object']).columns
for col in object_cols:
    df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
    df[col] = df[col].astype(str)

# Separate Target
target_col = 'is_fraudulent'
if target_col not in df.columns:
    print(f"❌ Target column '{target_col}' not found!")
    exit()

X = df.drop(columns=[target_col])
y = df[target_col]

# Save training data for app to use (hospital list etc)
print("💾 Saving training_data.csv for app...")
df.to_csv('training_data.csv', index=False)

# 3. Setup Pipeline
print("⚙️ Setting up pipelines...")
categorical_cols = X.select_dtypes(include='object').columns
numeric_cols = X.select_dtypes(exclude='object').columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ]
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Save feature columns
print("📝 Saving feature columns...")
with open('model_columns.pkl', 'wb') as f:
    pickle.dump(X_train.columns.tolist(), f)

# Save preprocessor
# Note: We need to fit the preprocessor first
X_train_processed = preprocessor.fit_transform(X_train)
with open('preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)

# 4. Train Models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Isolation Forest': IsolationForest(random_state=42, contamination=0.1)
}

smote = SMOTE(random_state=42)

for name, model in models.items():
    print(f"🏋️ Training {name}...")
    try:
        if name == 'Isolation Forest':
            # Isolation Forest is unsupervised, use X_train only (processed)
            # It detects anomalies. We assume fraud cases are anomalies.
            model.fit(X_train_processed)
            save_name = 'isolation_forest.pkl'
        else:
            # Classification models
            # Use pipeline with SMOTE
            # Need to separate preprocessor because app loads it separately
            
            # Recalculate X_train_resampled for manual training
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)
            
            model.fit(X_train_resampled, y_train_resampled)
            
            if name == 'Random Forest': save_name = 'random_forest.pkl'
            elif name == 'Logistic Regression': save_name = 'logistic_regression.pkl'
            elif name == 'Decision Tree': save_name = 'decision_tree.pkl'
            elif name == 'KNN': save_name = 'knn.pkl'
            
        with open(save_name, 'wb') as f:
            pickle.dump(model, f)
        print(f"   ✅ Saved {save_name}")
        
    except Exception as e:
        print(f"   ❌ Failed to train {name}: {e}")

print("✅ Done! Models regenerated.")
