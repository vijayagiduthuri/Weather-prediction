import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
df = pd.read_csv("weather.csv")
display(df.head())
x = df.columns
display(x)
unnecessary_columns = [
    'STATION', 'NAME', 'AWND_ATTRIBUTES', 'PGTM_ATTRIBUTES', 'PRCP_ATTRIBUTES', 
    'SNOW_ATTRIBUTES', 'SNWD_ATTRIBUTES', 'TAVG_ATTRIBUTES', 'TMAX_ATTRIBUTES', 
    'TMIN_ATTRIBUTES', 'WDF2_ATTRIBUTES', 'WDF5_ATTRIBUTES', 'WSF2_ATTRIBUTES', 
    'WSF5_ATTRIBUTES', 'WT01_ATTRIBUTES', 'WT02_ATTRIBUTES', 'WT03_ATTRIBUTES', 
    'WT04_ATTRIBUTES', 'WT05_ATTRIBUTES', 'WT06_ATTRIBUTES', 'WT08_ATTRIBUTES', 
    'WT09_ATTRIBUTES'
]
df = df.drop(columns=unnecessary_columns, errors='ignore')
display(df.head())
selected_columns = ['LATITUDE', 'LONGITUDE', 'ELEVATION', 'AWND', 'PRCP', 'SNOW', 
                    'SNWD', 'TAVG', 'TMAX', 'TMIN','DATE']
df_cleaned = df[selected_columns]
df_cleaned.to_csv("cleaned_weather_data.csv", index=False)
display(df_cleaned)
df_cleaned = df_cleaned.dropna(subset=['TAVG'])
display(df_cleaned)
# Get only numeric columns
numeric_cols = df_cleaned.select_dtypes(include='number').columns
# Fill NaNs in numeric columns with their median values
df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(df_cleaned[numeric_cols].median())
display(df_cleaned)
df_cleaned['DATE'] = pd.to_datetime(df_cleaned['DATE'])
df_cleaned['Year'] = df_cleaned['DATE'].dt.year
df_cleaned['Month'] = df_cleaned['DATE'].dt.month
df_cleaned['Day'] = df_cleaned['DATE'].dt.day
display(df_cleaned.head())
df_cleaned = df_cleaned.drop(columns=['DATE'])
display(df_cleaned.head())
X = df_cleaned.drop('TAVG', axis=1)
y = df_cleaned['TAVG']
display(X.head())
display(y.head())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
date_subset = X_test.iloc[:15].copy()
date_subset['TAVG'] = y_test.iloc[:15].values

timestamps = [f"{int(row['Year'])}-{int(row['Month']):02d}-{int(row['Day']):02d}" for _, row in date_subset.iterrows()]
display(timestamps)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# Initialize and train the model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict
rf_pred = rf_model.predict(X_test)

# Evaluate
rf_mse = mean_squared_error(y_test, rf_pred)
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)

print("Random Forest Regressor:")
print(f"  Accuracy (R²) = {rf_r2 * 100:.2f}%")
print(f"  MSE = {rf_mse:.2f}")
print(f"  MAE = {rf_mae:.2f}\n")

rf_pred_subset = rf_pred[:15]

plt.figure(figsize=(10, 5))
plt.plot(timestamps, y_test_subset, label='Actual TAVG', marker='o', color='blue')
plt.plot(timestamps, rf_pred_subset, label='Predicted TAVG', marker='o', color='orange')
plt.title('Random Forest: Actual vs Predicted (First 15 Samples)')
plt.xlabel('Date')
plt.ylabel('TAVG')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

from sklearn.neighbors import KNeighborsRegressor

# Initialize and train the model
knn_model = KNeighborsRegressor()
knn_model.fit(X_train, y_train)

# Predict
knn_pred = knn_model.predict(X_test)

# Evaluate
knn_mse = mean_squared_error(y_test, knn_pred)
knn_mae = mean_absolute_error(y_test, knn_pred)
knn_r2 = r2_score(y_test, knn_pred)

print("K-Nearest Neighbors Regressor:")
print(f"  Accuracy (R²) = {knn_r2 * 100:.2f}%")
print(f"  MSE = {knn_mse:.2f}")
print(f"  MAE = {knn_mae:.2f}\n")

knn_pred_subset = knn_pred[:15]

plt.figure(figsize=(10, 5))
plt.plot(timestamps, y_test_subset, label='Actual TAVG', marker='o', color='blue')
plt.plot(timestamps, knn_pred_subset, label='Predicted TAVG', marker='o', color='orange')
plt.title('K-Nearest Neighbors: Actual vs Predicted (First 15 Samples)')
plt.xlabel('Date')
plt.ylabel('TAVG')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
from sklearn.linear_model import LinearRegression


# Initialize and train the model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predict
lr_pred = lr_model.predict(X_test)

# Evaluate
lr_mse = mean_squared_error(y_test, lr_pred)
lr_mae = mean_absolute_error(y_test, lr_pred)
lr_r2 = r2_score(y_test, lr_pred)

print("Linear Regression:")
print(f"  Accuracy (R²) = {lr_r2 * 100:.2f}%")
print(f"  MSE = {lr_mse:.2f}")
print(f"  MAE = {lr_mae:.2f}\n")

# Predictions
y_test_subset = y_test.iloc[:15].values
lr_pred_subset = lr_pred[:15]

# Plot
plt.figure(figsize=(10, 5))
plt.plot(timestamps, y_test_subset, label='Actual TAVG', marker='o', color='blue')
plt.plot(timestamps, lr_pred_subset, label='Predicted TAVG', marker='o', color='orange')
plt.title('Linear Regression: Actual vs Predicted (First 15 Samples)')
plt.xlabel('Date')
plt.ylabel('TAVG')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

from sklearn.svm import SVR

# Initialize and train the model
svr_model = SVR()
svr_model.fit(X_train, y_train)

# Predict
svr_pred = svr_model.predict(X_test)

# Evaluate
svr_mse = mean_squared_error(y_test, svr_pred)
svr_mae = mean_absolute_error(y_test, svr_pred)
svr_r2 = r2_score(y_test, svr_pred)

print("Support Vector Regressor:")
print(f"  Accuracy (R²) = {svr_r2 * 100:.2f}%")
print(f"  MSE = {svr_mse:.2f}")
print(f"  MAE = {svr_mae:.2f}\n")

svm_pred_subset = svr_pred[:15]

plt.figure(figsize=(10, 5))
plt.plot(timestamps, y_test_subset, label='Actual TAVG', marker='o', color='blue')
plt.plot(timestamps, svm_pred_subset, label='Predicted TAVG', marker='o', color='orange')
plt.title('Support Vector Machine: Actual vs Predicted (First 15 Samples)')
plt.xlabel('Date')
plt.ylabel('TAVG')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
from sklearn.linear_model import LinearRegression
import numpy as np
print("Training on features:", X.columns) 
# Retrain the model
best_model = LinearRegression()
best_model.fit(X, y)  # Important!

columns = ['LATITUDE', 'LONGITUDE', 'ELEVATION', 'AWND', 'PRCP', 'SNOW',
           'SNWD', 'TMAX', 'TMIN', 'Year', 'Month', 'Day']

sunny_input = [[48.55943, -93.39554, 352.6, 8.6,
                         68.1, 325.0, 660.0, 36.7, 21.7,
                         2025, 12, 31]]
rainy_input = [[48.55943, -93.39554, 352.6, 4.5,
                         12.0, 0.0, 0.0, 22.0, 18.0,
                         2025, 7, 10]]
cold_input = [[48.55943, -93.39554, 352.6, 3.0,
                         0.0, 20.0, 25.0, -5.0, -15.0,
                         2025, 1, 5]]
def temp_emoji(temp):
    if temp >= 25:
        return "☀️"
    elif temp <= 10:
        return "❄️"
    else:
        return "🌧️"
# Predict
sunny_input_df = pd.DataFrame(sunny_input, columns=columns)
rainy_input_df = pd.DataFrame(rainy_input, columns=columns)
cold_input_df = pd.DataFrame(cold_input, columns=columns)
for label, input_array in zip(['Sunny', 'Rainy', 'Cold'],[sunny_input_df, rainy_input_df, cold_input_df]):
    predicted_tavg = best_model.predict(input_array)[0]
    print(f"{label} Day Prediction: {predicted_tavg:.2f}°C {temp_emoji(predicted_tavg)}")