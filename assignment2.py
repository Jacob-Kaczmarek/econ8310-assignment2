# assignment2.py

import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import OrdinalEncoder

# 1. Load data
train_url = "https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv"
test_url = "https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv"

train_df = pd.read_csv(train_url)
test_df = pd.read_csv(test_url)

# 2. Prepare the data
y = train_df['meal']
X = train_df.drop(columns=['meal'])

# 3. Encode categorical variables safely
categorical_cols = X.select_dtypes(include='object').columns.tolist()

encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X[categorical_cols] = encoder.fit_transform(X[categorical_cols])
test_df[categorical_cols] = encoder.transform(test_df[categorical_cols])

# Make sure test has the same features
X_test = test_df[X.columns]

# 4. Define model
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

# 5. Fit model
modelFit = model.fit(X, y)

# 6. Predict on test set
pred = modelFit.predict(X_test)
pred = pred.astype(int)
