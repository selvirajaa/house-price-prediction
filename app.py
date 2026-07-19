# App.py
import pandas as pd
import streamlit as st
import joblib
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

# Load Dataset
df = pd.read_csv("house_price_advanced_12000_plus.csv")
print(df.head())
# Data Cleaning
print(df.info())
print(df.shape)
print(df.columns)
print(df.describe())
print(df.isnull().sum())
df.fillna(df.mean(numeric_only = True), inplace = True)
df["Furnishing"] = df["Furnishing"].fillna(df["Furnishing"].mode()[0])
print(df["Furnishing"].isnull().sum())
print(df.duplicated().sum())
df.drop_duplicates(inplace =True)
print(df["Location"].unique())
print(df["Furnishing"].unique())
print(df["Electricity_Availability"].unique())
print(df["Water_Supply"].unique())
print(df["Security"].unique())
print(df["House_Type"].unique())
print(df["Facing"].unique())


# features and target
x = df[['Area',
        'Bedrooms',
        'Bathrooms',
        'Floors',
        'Age',
        'Parking',
        'Location',
        'Furnishing',
        'Distance_to_City',
        'Crime_Rate',
        'School_Rating',
        'Hospital_Distance',
        'Metro_Distance',
        'Shopping_Mall_Distance',
        'Park_Distance',
        'Pollution_Index',
        'Population_Density',
        'Property_Tax',
        'Electricity_Availability',
        'Water_Supply',
        'Security',
        'House_Type',
        'Facing',
       ]]
y = df[['Price']]
# convert categorical columns
x = pd.get_dummies(x, drop_first = True)
# save training columns
joblib.dump(x.columns.tolist(), 'model_columns_12000.pkl')
# Train Test Split
x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size = 0.20,
                                                    random_state = 42)
model = LinearRegression()
model.fit(x_train, y_train)

# Prediction
y_pred = model.predict(x_test)
residuals = y_test - y_pred
# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse =np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2_Score:",r2)

# save model
joblib.dump(model, "linear_model_12000.pkl")
print("\nmodel saved successfully")


# Page Configuration
st.set_page_config(
    page_title = "House Price Prediction",
    page_icon = "🏠",
    layout = "centered"
)

# load models
model = joblib.load("linear_model_12000.pkl")
model_columns = joblib.load("model_columns_12000.pkl")
try:
    model = joblib.load("linear_model_12000.pkl")
    model_columns = joblib.load("model_columns_12000.pkl")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Title
st.title("🏠House Price Prediction System")
st.markdown("""
### Welcome!

This application predicts the estimated house price using a Machine Learning Linear Regression model.

Enter the house details below and click **Predict Price**.
""")
st.subheader("📊 Model Performance")

st.write(f"R² Score : {r2:.4f}")
st.write(f"MAE : {mae:.2f}")
st.write(f"RMSE : {rmse:.2f}")

# Input fields
col1, col2 = st.columns(2)
with col1:
    area = st.number_input("Area(sq.ft)",
                            min_value = 300,
                            value =1000,
                            step = 10)
    bedrooms = st.number_input("Bedrooms",
                                min_value =1.0,
                                value = 2.0,
                                step = 0.5)
    bathrooms = st.number_input("Bathrooms",
                                min_value = 1,
                                value = 2,
                                step = 1)
    floors = st.number_input("Floors",
                             min_value = 1,
                             value = 1,
                             step = 1)
    age = st.number_input("Age",
                          min_value = 0.0,
                          value = 5.0,
                          step = 0.1)
    parking = st.number_input("Parking",
                              min_value = 0,
                              value = 1,
                              step = 1)
with col2:
    distance_to_city = st.number_input("Distance_to_City",
                                       min_value = 0.0,
                                       value =5.0,
                                       step =0.10)
    crime_rate = st.number_input("Crime_Rate",
                                 min_value = 0.0,
                                 value = 2.5,
                                 step = 0.1)
    school_rating = st.number_input("School_Rating",
                                    min_value = 1,
                                    max_value = 10,
                                    value = 7,
                                    step = 1)
    hospital_distance = st.number_input("Hospital_Distance",
                                        min_value = 0.0,
                                        value =5.0,
                                        step = 0.1)
    metro_distance = st.number_input("Metro_Distance",
                                     min_value = 0.0,
                                     value =5.0,
                                     step = 0.1)
    shopping_mall_distance = st.number_input("Shopping_Mall_Distance",
                                             min_value = 0.0,
                                             value = 3.0,
                                             step = 0.1)
    park_distance = st.number_input("Park_Distance",
                                    min_value = 0.0,
                                    value = 0.5,
                                    step = 0.1)
    population_density = st.number_input("Population_Density",
                                         min_value = 0,
                                         value = 30,
                                         step = 1)
    pollution_index = st.number_input("Pollution_Index",
                                      min_value=0.0,
                                      value=50.0,
                                      step=1.0)
    property_tax = st.number_input("Property_Tax",
                                   min_value=0.0,
                                   value=5000.0,
                                   step=100.0)

    # Dropdown menu
    location = st.selectbox("Location",
                            ["Urban", "Suburban", "Rural"]
                            )
    furnishing = st.selectbox("Furnishing",
                              ["Furnished", "Semi-Furnished", "Unfurnished"]
                              )
    electricity_availability = st.selectbox("Electricity_Availability",
                                            ["Yes", "No"]
                                            )
    water_supply = st.selectbox("Water_Supply",
                                ["24x7", "Limited"]
                                )
    house_type = st.selectbox("House_Type",
                              ["Apartment", "Independent House", "Villa"]
                              )
    security = st.selectbox("Security",
                            ["Gated", "Non-Gated"]
                            )
    facing = st.selectbox("Facing",
                          ["West", "East", "North", "South"]
                          )
    st.markdown("-------")


 # Predict Button
    if st.button("Predict Price"):
        with st.spinner("Predicting House Price....."):
            input_df = pd.DataFrame({
                "Area": [area],
                "Bedrooms": [bedrooms],
                "Bathrooms": [bathrooms],
                "Floors": [floors],
                "Age": [age],
                "Parking": [parking],
                "Location": [location],
                "Furnishing": [furnishing],
                "Distance_to_City": [distance_to_city],
                "Crime_Rate": [crime_rate],
                "School_Rating": [school_rating],
                "Hospital_Distance": [hospital_distance],
                "Metro_Distance": [metro_distance],
                "Shopping_Mall_Distance": [shopping_mall_distance],
                "Park_Distance": [park_distance],
                "Pollution_Index": [pollution_index],
                "Population_Density": [population_density],
                "Property_Tax": [property_tax],
                "Electricity_Availability": [electricity_availability],
                "Water_Supply": [water_supply],
                "Security": [security],
                "House_Type": [house_type],
                "Facing": [facing]
            })
            # convert categorical columns
            input_df = pd.get_dummies(input_df)
            # match training columns
            input_df = input_df.reindex(
                columns= model_columns,
                fill_value=0
            )
            # prediction
            prediction = model.predict(input_df)
            price = float(prediction.flatten()[0])
            # Show result
            st.balloons()
            st.success(f"🏠 Estimated House Price : {price:,.2f} Lakhs")
            st.markdown('------')
            # Display input Details
            st.subheader("Entered House Details")
            st.write({
                "Area": area,
                "Bedrooms": bedrooms,
                "Bathrooms": bathrooms,
                "Floors": floors,
                "Age": age,
                "Parking": parking,
                "Location": location,
                "Furnishing": furnishing,
                "Distance_to_City": distance_to_city,
                "Crime_Rate": crime_rate,
                "School_Rating": school_rating,
                "Hospital_Distance": hospital_distance,
                "Metro_Distance": metro_distance,
                "Shopping_Mall_Distance": shopping_mall_distance,
                "Park_Distance": park_distance,
                "Pollution_Index": pollution_index,
                "Population_Density": population_density,
                "Property_Tax": property_tax,
                "Electricity_Availability": electricity_availability,
                "Water_Supply": water_supply,
                "Security": security,
                "House_Type": house_type,
                "Facing": facing
            })

            st.subheader("Area vs House Price")

            st.scatter_chart(data=df, x="Area", y="Price")

            fig, ax = plt.subplots()

            ax.scatter(y_pred, residuals)
            ax.axhline(y=0, color="red")

            st.pyplot(fig)

            # footer
            st.markdown("--------")
            st.caption("Created Using Python, scikit_learn and streamlit")


