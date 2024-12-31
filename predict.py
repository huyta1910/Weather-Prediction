import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
file_path = r"D:\Data fundermental\Data Mining\weather prediction\Weather HCMC2020To8-2024 Full.csv"
weather_data = pd.read_csv(file_path)

# Display initial information about the dataset
print("Initial dataset info:")
print(weather_data.info())

# Rename columns for consistency (remove Vietnamese and special characters)
weather_data.columns = [
    "Date", "TempMax", "TempMin", "TempAvg", "FeelsLikeMax", "FeelsLikeMin", "FeelsLikeAvg",
    "DewPoint", "Humidity", "WindGust", "WindSpeed", "WindDir", "SeaLevel", "CloudCover",
    "Visibility", "SolarRadiation", "SolarEnergy", "UVIndex", "Sunrise", "Sunset",
    "MoonPhase", "Conditions", "Description", "RainToday"
]

# Convert Date to datetime and drop rows with invalid dates
print("Number of rows before dropping invalid dates:", len(weather_data))
weather_data['Date'] = pd.to_datetime(weather_data['Date'], errors='coerce')
weather_data = weather_data.dropna(subset=['Date'])
print("Number of rows after dropping invalid dates:", len(weather_data))

# Select relevant columns and convert them to numeric if applicable
columns_to_convert = [
    "TempMax", "TempMin", "TempAvg", "FeelsLikeMax", "FeelsLikeMin", "FeelsLikeAvg",
    "DewPoint", "Humidity", "WindGust", "WindSpeed", "Visibility", "SolarRadiation",
    "SolarEnergy", "UVIndex"
]

for col in columns_to_convert:
    weather_data[col] = pd.to_numeric(weather_data[col], errors='coerce')
    # Fill missing values with the column mean
    weather_data[col] = weather_data[col].fillna(weather_data[col].mean())

print("Number of rows after handling numeric columns:", len(weather_data))

# Handle the target variable 'RainToday'
weather_data['RainToday'] = weather_data['RainToday'].apply(lambda x: 1 if pd.notnull(x) else 0)

# Create 'RainTomorrow' target by shifting 'RainToday' and drop the last row
weather_data['RainTomorrow'] = weather_data['RainToday'].shift(-1)
weather_data = weather_data[:-1]  # Drop last row since RainTomorrow is NaN

print("Number of rows before train-test split:", len(weather_data))

# Define features (X) and target (y)
features = [
    "TempMax", "TempMin", "TempAvg", "FeelsLikeMax", "FeelsLikeMin", "FeelsLikeAvg",
    "DewPoint", "Humidity", "WindGust", "WindSpeed", "Visibility", "SolarRadiation",
    "SolarEnergy", "UVIndex"
]
X = weather_data[features]
y = weather_data["RainTomorrow"].astype(int)

# Check final dataset shape
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

# Ensure there are samples before splitting
if len(X) == 0 or len(y) == 0:
    raise ValueError("Dataset is empty after preprocessing. Please review data cleaning steps.")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate the model
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, target_names=["No Rain", "Rain"])

# Print evaluation results
print("Model accuracy:", accuracy)
print("\nClassification report:\n", classification_rep)
