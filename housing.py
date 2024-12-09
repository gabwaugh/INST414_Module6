import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
file_path = 'Housing.csv'
housing_data = pd.read_csv(file_path)

# Explore the dataset
print(housing_data.head())
print(housing_data.info())
print(housing_data.describe())

# Preprocess the data
# Handle missing values (if any)
housing_data = housing_data.dropna()

# Encode categorical variables
categorical_columns = housing_data.select_dtypes(include=['object']).columns
housing_data = pd.get_dummies(housing_data, columns=categorical_columns, drop_first=True)

# Define target (price) and features
target = 'price'
X = housing_data.drop(columns=[target])
y = housing_data[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Regression Model
# Initialize the model
model = RandomForestRegressor(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the Model
# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R2): {r2}")

# Analyze Incorrect Predictions
# Compare true vs predicted values
comparison_df = pd.DataFrame({'True': y_test, 'Predicted': y_pred})
comparison_df['Error'] = abs(comparison_df['True'] - comparison_df['Predicted'])

# Identify Incorrect SamplesPredictions
incorrect_samples = comparison_df.sort_values(by='Error', ascending=False).head(5)
print("Top 5 incorrect predictions:")
print(incorrect_samples)

# Visualize Results
# Plot true vs predicted values
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("True Price")
plt.ylabel("Predicted Price")
plt.title("True vs Predicted Price")
plt.show()

# Feature importance
feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
feature_importances.plot(kind='bar', figsize=(10, 5), title="Feature Importances")
plt.show()