import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import requests

# Load historical weather data
weather_data = pd.read_csv(r'C:\Users\Dell\Desktop\weather_data.csv', skipinitialspace=True, sep='\t')

# Preprocess data: handle missing values, convert data types, etc.
print(weather_data.columns)  # Check the column names

# Fill missing values with the mean of the respective columns
weather_data['Temperature(°C)'] = weather_data['Temperature(°C)'].fillna(weather_data['Temperature(°C)'].mean())
weather_data['Humidity (%)'] = weather_data['Humidity (%)'].fillna(weather_data['Humidity (%)'].mean())
weather_data['Wind Speed (km/h)'] = weather_data['Wind Speed (km/h)'].fillna(weather_data['Wind Speed (km/h)'].mean())

# Fill missing values in the target variable with the most frequent value
weather_data['Heat Wave'] = weather_data['Heat Wave'].fillna(weather_data['Heat Wave'].mode()[0])

# Convert 'Date' column to datetime format (assuming this is the correct column name)
weather_data['Date'] = pd.to_datetime(weather_data['Date'])  # Convert 'Date' column to datetime format

weather_data.set_index('Date', inplace=True)

# Split data into training and testing sets (80% for training, 20% for testing)
X = weather_data[['Temperature(°C)', 'Humidity (%)', 'Wind Speed (km/h)']]
y = weather_data['Heat Wave']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a random forest classifier on the training data
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, y_train)

# Evaluate the model using the testing data
y_pred = rfc.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save the trained model to a file
joblib.dump(rfc, 'heat_wave_model.joblib')

# Function to predict heat waves for a given set of weather conditions
def predict_heat_wave(temperature, humidity, wind_speed):
    # Preprocess input data
    input_data = pd.DataFrame({'Temperature(°C)': [temperature], 'Humidity (%)': [humidity], 'Wind Speed (km/h)': [wind_speed]})

    # Predict the likelihood of a heat wave using the trained model
    prediction = rfc.predict(input_data)[0]

    return prediction

# Example usage:
temperature = 35  # degrees Celsius
humidity = 60  # percent
wind_speed = 10  # km/h

prediction = predict_heat_wave(temperature, humidity, wind_speed)
if prediction == 1:
    print("Heat wave predicted!")
else:
    print("No heat wave predicted.")

# Send an alert to users through mobile platforms (e.g., SMS or push notifications)
def send_alert():
    # Replace with your preferred mobile platform API or service
    url = "https://your-mobile-platform-api.com/send-alert"
    data = {"message": "Heat wave alert! Stay cool and hydrated."}
    response = requests.post(url, json=data)
    if response.status_code == 200:
        print("Alert sent successfully!")
    else:
        print("Error sending alert:", response.text)

# Trigger the alert if a heat wave is predicted
if prediction == 1:
    send_alert()