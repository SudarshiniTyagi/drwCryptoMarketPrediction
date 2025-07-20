import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor, plot_importance
from scipy.stats import pearsonr

train_data = pd.read_parquet('data/train.parquet')
# test_data = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/test.parquet')
# print(train_data.head())

# Suppress the RuntimeWarning
np.seterr(invalid='ignore')

# Clip and plot the label distribution
plot_labels_dist(train_data)

#Add lag_1 and lag_2
train_data = create_more_features(train_data)

# Select key features ('bid_qty', 'ask_qty', 'volume', 'lag_1', 'label') and remove rows with NaN values (from the first lag)
# X = train_data[['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume', 'X1','X2','X3',
#                 'X4','X5','lag_1','lag_2','label']].dropna()
X = train_data.dropna()

# Extract the 'label' column from X as the target variable y, removing it from the feature set
y = X.pop('label')
# timestamp = X.pop('timestamp')
# Display the first few rows of the feature set X to confirm the structure
print(X.head())

# Display the first few values of the target variable y to verify it matches X
print(y.head())



X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

model = XGBRegressor()
model.fit(X_train, y_train)

predictions = model.predict(X_val)
corr, _ = pearsonr(predictions, y_val)
print("Pearson correlation:", corr)

# --- Get feature importances as numbers ---
importance_scores = model.feature_importances_

# Combine into a dataframe for easier viewing
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importance_scores
}).sort_values(by='Importance', ascending=False)

print(feature_importance_df)

# --- Plot feature importance ---
plt.figure(figsize=(10, 6))
plot_importance(model, importance_type='weight', max_num_features=10, height=0.5)
plt.title("Top 10 Feature Importances")
plt.show()