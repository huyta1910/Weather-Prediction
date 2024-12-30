import pandas as pd

# Load the dataset to inspect its structure and contents
file_path = r"D:\Data fundermental\Data Mining\weather prediction\Weather HCMC2020To8-2024 Full.csv"
weather_data = pd.read_csv(file_path)

# Display the first few rows of the dataset to understand its structure
weather_data.head(), weather_data.info()
# Preprocessing the dataset

# Rename columns for consistency (remove Vietnamese and special characters)
weather_data.columns = [
    "Date", "TempMax", "TempMin", "TempAvg", "FeelsLikeMax", "FeelsLikeMin", "FeelsLikeAvg",
    "DewPoint", "Humidity", "WindGust", "WindSpeed", "WindDir", "SeaLevel", "CloudCover",
    "Visibility", "SolarRadiation", "SolarEnergy", "UVIndex", "Sunrise", "Sunset",
    "MoonPhase", "Conditions", "Description", "RainToday"
]

# Convert Date to datetime
weather_data['Date'] = pd.to_datetime(weather_data['Date'], errors='coerce')

# Drop rows with invalid dates
weather_data = weather_data.dropna(subset=['Date'])

# Select relevant columns and convert them to numeric if applicable
columns_to_convert = [
    "TempMax", "TempMin", "TempAvg", "FeelsLikeMax", "FeelsLikeMin", "FeelsLikeAvg",
    "DewPoint", "Humidity", "WindGust", "WindSpeed", "Visibility", "SolarRadiation",
    "SolarEnergy", "UVIndex"
]

for col in columns_to_convert:
    weather_data[col] = pd.to_numeric(weather_data[col], errors='coerce')

# Drop rows with missing values in important numeric columns
weather_data = weather_data.dropna(subset=columns_to_convert)

# Handle the target variable 'RainToday' (assume 'rain' if not null, otherwise 'no rain')
weather_data['RainToday'] = weather_data['RainToday'].apply(lambda x: 1 if pd.notnull(x) else 0)

# Create a "RainTomorrow" target by shifting 'RainToday' column
weather_data['RainTomorrow'] = weather_data['RainToday'].shift(-1)

# Drop the last row (no data for tomorrow's rain)
weather_data = weather_data[:-1]

# Preview the cleaned dataset
weather_data.head()

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Define features (X) and target (y)
features = [
    "TempMax", "TempMin", "TempAvg", "FeelsLikeMax", "FeelsLikeMin", "FeelsLikeAvg",
    "DewPoint", "Humidity", "WindGust", "WindSpeed", "Visibility", "SolarRadiation",
    "SolarEnergy", "UVIndex"
]
X = weather_data[features]
y = weather_data["RainTomorrow"].astype(int)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate the model
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, target_names=["No Rain", "Rain"])

accuracy, classification_rep
