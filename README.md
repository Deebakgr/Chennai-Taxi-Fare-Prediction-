# How It Works
Load Data: A pre-processed dataset of Chennai locations and pin codes is used for model training.
Model Training: The Random Forest Regressor is trained using trip data, including distance, duration, time of day, and traffic conditions.
# User Inputs:
Select the From Place and To Place from a dropdown menu.
Enter the Distance and Duration manually.
Choose the Traffic Condition based on real-time observations.
# Fare Prediction: The app uses a combination of:
Distance-based pricing.
Traffic and time multipliers.
Predicted fare displayed instantly.
# Technologies Used
Python: Core programming language.
Streamlit: For building the interactive web application.
Scikit-learn: Machine learning library for model training and prediction.
Pandas and NumPy: For data handling and manipulation.
Google Maps Integration: Dynamically generate maps and routes (manual inputs for distance and duration).



