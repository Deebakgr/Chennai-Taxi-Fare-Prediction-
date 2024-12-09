


# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from datetime import datetime

# # Load datasets
# @st.cache_data
# def load_chennai_pincode_data(file_path):
#     return pd.read_csv(file_path)

# # Train RandomForest model
# @st.cache_resource
# def train_model(data):
#     features = data[['distance', 'duration', 'time_of_day', 'traffic_condition']]
#     target = data['fare']

#     X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)

#     model = RandomForestRegressor(random_state=42, n_estimators=100, max_depth=10)
#     model.fit(X_train_scaled, y_train)

#     predictions = model.predict(X_test_scaled)
#     rmse = np.sqrt(mean_squared_error(y_test, predictions))
#     r2 = r2_score(y_test, predictions)

#     st.write(f"Model Evaluation: RMSE={rmse:.2f}, R2={r2:.2f}")
#     return model, scaler

# # Map hour to time of day
# def map_time_of_day(hour):
#     if 6 <= hour < 12:
#         return "Morning"
#     elif 12 <= hour < 17:
#         return "Afternoon"
#     elif 17 <= hour < 21:
#         return "Evening"
#     else:
#         return "Night"

# # Streamlit app
# st.set_page_config(page_title="Chennai Taxi Fare Predictor", page_icon="🚖")

# # Load Chennai pincode dataset
# chennai_pincode_df = load_chennai_pincode_data('Chennai_pincode.csv')

# # Sample data for training (replace with actual data loading)
# @st.cache_data
# def load_sample_data():
#     np.random.seed(42)
#     sample_data = pd.DataFrame({
#         'distance': np.random.uniform(2, 25, 500),
#         'duration': np.random.uniform(5, 60, 500),
#         'time_of_day': np.random.choice(['Morning', 'Afternoon', 'Evening', 'Night'], 500),
#         'traffic_condition': np.random.choice(['Light Traffic', 'Moderate Traffic', 'Heavy Traffic'], 500),
#         'fare': np.random.uniform(50, 500, 500)
#     })

#     # Map categorical values to numerical
#     time_mapping = {'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3}
#     traffic_mapping = {'Light Traffic': 0, 'Moderate Traffic': 1, 'Heavy Traffic': 2}

#     sample_data['time_of_day'] = sample_data['time_of_day'].map(time_mapping)
#     sample_data['traffic_condition'] = sample_data['traffic_condition'].map(traffic_mapping)
#     return sample_data

# data = load_sample_data()
# model, scaler = train_model(data)

# st.subheader('Predict Fare from Place to Place')

# # Input: From and To Place
# from_place = st.selectbox('From Place', chennai_pincode_df['place_name'].unique(), key='from_place')
# to_place = st.selectbox('To Place', chennai_pincode_df['place_name'].unique(), key='to_place')

# from_location = chennai_pincode_df[chennai_pincode_df['place_name'] == from_place][['latitude', 'longitude']].values
# to_location = chennai_pincode_df[chennai_pincode_df['place_name'] == to_place][['latitude', 'longitude']].values

# if from_location.size > 0 and to_location.size > 0:
#     from_location = from_location[0]
#     to_location = to_location[0]

#     # Generate Google Maps URL
#     google_maps_url = f"https://www.google.com/maps/dir/?api=1&origin={from_location[0]},{from_location[1]}&destination={to_location[0]},{to_location[1]}&travelmode=driving"

#     # Display link to Google Maps
#     st.markdown(f'[View Route on Google Maps]({google_maps_url})')

#     st.write("Please open the link above, check the shortest distance and estimated duration on Google Maps, and enter them below.")

#     # Manual input for distance and duration
#     distance = st.number_input('Enter the distance (in km)', min_value=0.0, format="%.2f")
#     duration = st.number_input('Enter the duration (in minutes)', min_value=0.0, format="%.2f")

#     # Allow the user to select traffic condition
#     traffic_condition = st.radio(
#         "Select Traffic Condition",
#         ('Light Traffic', 'Moderate Traffic', 'Heavy Traffic'),
#         index=1  # Default to 'Moderate Traffic'
#     )

#     # Determine time of day
#     current_hour = datetime.now().hour
#     time_of_day = map_time_of_day(current_hour)

#     time_mapping = {'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3}
#     traffic_mapping = {'Light Traffic': 0.9, 'Moderate Traffic': 1.0, 'Heavy Traffic': 1.2}

#     if distance > 0 and duration > 0:
#         st.write(f"Entered Distance: {distance:.2f} km")
#         st.write(f"Entered Duration: {duration:.2f} mins")
#         st.write(f"Time of Day: {time_of_day}")
#         st.write(f"Selected Traffic Condition: {traffic_condition}")

#         if st.button('Predict Fare'):
#             base_fare = distance * 10 + duration * 2  # Base fare logic
#             traffic_multiplier = traffic_mapping[traffic_condition]
#             time_multiplier = 1.1 if time_of_day in ["Evening", "Night"] else 1.0

#             adjusted_fare = base_fare * traffic_multiplier * time_multiplier
#             st.success(f"Predicted Fare from {from_place} to {to_place}: ₹{adjusted_fare:.2f}")
#     else:
#         st.warning("Please ensure both distance and duration are greater than zero.")
# else:
#     st.error("Unable to fetch location data for the selected places.")



















import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime

# Load datasets
@st.cache_data
def load_chennai_pincode_data(file_path):
    return pd.read_csv(file_path)

# Train RandomForest model
@st.cache_resource
def train_model(data):
    features = data[['distance', 'duration', 'time_of_day', 'traffic_condition']]
    target = data['fare']

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestRegressor(random_state=42, n_estimators=100, max_depth=10)
    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    return model, scaler

# Map hour to time of day
def map_time_of_day(hour):
    if 6 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 21:
        return "Evening"
    else:
        return "Night"

# Streamlit app
st.set_page_config(page_title="Chennai Taxi Fare Predictor", page_icon="🚖")

# Load Chennai pincode dataset
chennai_pincode_df = load_chennai_pincode_data('Chennai_pincode.csv')

# Sample data for training
@st.cache_data
def load_sample_data():
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'distance': np.random.uniform(2, 25, 500),
        'duration': np.random.uniform(5, 60, 500),
        'time_of_day': np.random.choice(['Morning', 'Afternoon', 'Evening', 'Night'], 500),
        'traffic_condition': np.random.choice(['Light Traffic', 'Moderate Traffic', 'Heavy Traffic'], 500),
        'fare': np.random.uniform(50, 500, 500)
    })

    # Map categorical values to numerical
    time_mapping = {'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3}
    traffic_mapping = {'Light Traffic': 0, 'Moderate Traffic': 1, 'Heavy Traffic': 2}

    sample_data['time_of_day'] = sample_data['time_of_day'].map(time_mapping)
    sample_data['traffic_condition'] = sample_data['traffic_condition'].map(traffic_mapping)
    return sample_data

data = load_sample_data()
model, scaler = train_model(data)

st.subheader('Predict Fare from Place to Place')

# Input: From and To Place
from_place = st.selectbox('From Place', chennai_pincode_df['place_name'].unique(), key='from_place')
to_place = st.selectbox('To Place', chennai_pincode_df['place_name'].unique(), key='to_place')

from_location = chennai_pincode_df[chennai_pincode_df['place_name'] == from_place][['latitude', 'longitude']].values
to_location = chennai_pincode_df[chennai_pincode_df['place_name'] == to_place][['latitude', 'longitude']].values

if from_location.size > 0 and to_location.size > 0:
    from_location = from_location[0]
    to_location = to_location[0]

    # Generate Google Maps URL
    google_maps_url = f"https://www.google.com/maps/dir/?api=1&origin={from_location[0]},{from_location[1]}&destination={to_location[0]},{to_location[1]}&travelmode=driving"

    # Display link to Google Maps
    st.markdown(f'[View Route on Google Maps]({google_maps_url})')

    st.write("Please open the link above, check the shortest distance and estimated duration on Google Maps, and enter them below.")

    # Manual input for distance and duration
    distance = st.number_input('Enter the distance (in km)', min_value=0.0, format="%.2f")
    duration = st.number_input('Enter the duration (in minutes)', min_value=0.0, format="%.2f")

    # Allow the user to select traffic condition
    traffic_condition = st.radio(
        "Select Traffic Condition",
        ('Light Traffic', 'Moderate Traffic', 'Heavy Traffic'),
        index=1
    )

    # Determine time of day
    current_hour = datetime.now().hour
    time_of_day = map_time_of_day(current_hour)

    # Traffic and time multipliers
    traffic_mapping = {'Light Traffic': 0.7, 'Moderate Traffic': 0.9, 'Heavy Traffic': 1.0}
    time_mapping = {'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3}

    if distance > 0 and duration > 0:
        st.write(f"Entered Distance: {distance:.2f} km")
        st.write(f"Entered Duration: {duration:.2f} mins")
        st.write(f"Time of Day: {time_of_day}")
        st.write(f"Selected Traffic Condition: {traffic_condition}")

        if st.button('Predict Fare'):
            base_fare = (distance * 12) + (duration * 3) + 20  # Adjusted fare logic
            traffic_multiplier = traffic_mapping[traffic_condition]
            time_multiplier = 1.2 if time_of_day in ["Evening", "Night"] else 1.0

            adjusted_fare = base_fare * traffic_multiplier * time_multiplier
            st.success(f"Predicted Fare from {from_place} to {to_place}: ₹{adjusted_fare:.2f}")
    else:
        st.warning("Please ensure both distance and duration are greater than zero.")
else:
    st.error("Unable to fetch location data for the selected places.")
