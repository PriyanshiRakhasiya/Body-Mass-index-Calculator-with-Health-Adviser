import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

df = pd.read_csv("uploads/BMI_Health_Advice.csv") 

encoder = LabelEncoder()
df['Category'] = encoder.fit_transform(df['Category'])

gender_encoder = LabelEncoder()
df['Gender'] = gender_encoder.fit_transform(df['Gender']) 

# Features and target
X = df[['Height (m)', 'Weight (kg)', 'Age', 'Gender']]  
y = df['Category']

# Define K-Fold Cross-Validation
kf = KFold(n_splits=4, shuffle=True, random_state=42)  

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=kf, scoring='accuracy', n_jobs=-1)
grid_search.fit(X, y)

# Best model selection
best_model = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Accuracy: {grid_search.best_score_:.4f}")

with open("uploads/kfold_results.txt", "w") as file:
    file.write(f"Best Parameters: {grid_search.best_params_}\n")
    file.write(f"Best Accuracy: {grid_search.best_score_:.4f}\n")

best_model.fit(X, y)

# Save the trained model and encoders
if not os.path.exists('uploads'):
    os.makedirs('uploads')

joblib.dump(best_model, 'uploads/bmi_health_model.pkl')
joblib.dump(encoder, 'uploads/bmi_encoder.pkl')
joblib.dump(gender_encoder, 'uploads/gender_encoder.pkl')

print("Model and encoders saved as 'bmi_health_model.pkl', 'bmi_encoder.pkl', and 'gender_encoder.pkl'")
