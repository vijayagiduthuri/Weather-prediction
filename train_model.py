import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the cleaned data
df_cleaned = pd.read_csv("cleaned_weather_data.csv")

# Drop 'DayOfYear' if it exists
if 'DayOfYear' in df_cleaned.columns:
    df_cleaned = df_cleaned.drop(columns=['DayOfYear'])

# Define features (12 features only)
features = ['LATITUDE', 'LONGITUDE', 'ELEVATION', 'AWND', 'PRCP', 'SNOW',
            'SNWD', 'TMAX', 'TMIN', 'Year', 'Month', 'Day']

X = df_cleaned[features]
y = df_cleaned['TAVG']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

print("✅ Linear Regression model trained and saved as model.pkl")
