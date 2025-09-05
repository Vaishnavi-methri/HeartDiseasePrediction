import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import lightgbm as lgb

# Generate sample dataset
np.random.seed(42)
data = pd.DataFrame({
    'age': np.random.randint(30, 80, 100),
    'cholesterol': np.random.randint(150, 300, 100),
    'blood_pressure': np.random.randint(80, 180, 100),
    'disease': np.random.choice([0, 1], 100)
})

X = data[['age', 'cholesterol', 'blood_pressure']]
y = data['disease']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train LightGBM model
train_data = lgb.Dataset(X_train, label=y_train)
params = {'objective': 'binary', 'metric': 'binary_error'}
model = lgb.train(params, train_data, num_boost_round=50)

# Predict & evaluate
y_pred = model.predict(X_test)
y_pred_classes = [1 if p > 0.5 else 0 for p in y_pred]

print(f"Accuracy: {accuracy_score(y_test, y_pred_classes) * 100:.2f}%")