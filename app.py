import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import gdown
from surprise import SVD

model_id = '1OfTG1jyiM9VCND18TxbqADuMLYVJ9lEI'
model_path = 'model.pkl'

data_id = '12NtMnhiDcjWrQkowr5TzqgAy9CUmHJDm'
data_path = 'data.pkl'

model_perfomance_id = '1724JwhcHPOioJzNwxXFM03xYfz_NHp8_'
model_perfomance_path = 'comparison_df_set_1.csv'

# Function to download and load the model 
@st.cache_resource
def load_model():
    # Check if the model file already exists
    if os.path.exists(model_path):
        st.write("Model already exists. Skipping download.")
        
        model = joblib.load(model_path)

        return model

    # If not, download the model
    url = f"https://drive.google.com/uc?id={model_id}"
    st.write("Downloading model from Google Drive...")
    gdown.download(url, model_path, quiet=False)
    st.write("Model downloaded!")

    model = joblib.load(model_path)
    
    return model

# Function to download and load the datasets
def load_data():
    # Load Events Dataset
    # Check if the model file already exists
    if os.path.exists(data_path) and os.path.exists(model_perfomance_path):
        st.write("Data (1) and Data (2) already exist. Skipping download.")
        
        data = joblib.load(data_path)
        df = pd.read_csv(model_perfomance_path)
        df.drop(columns=['Unnamed: 0'], inplace=True)
        
        return df, data

    # If not, download the model
    url = f"https://drive.google.com/uc?id={data_id}"
    st.write("Downloading model from Google Drive...")
    gdown.download(url, data_path, quiet=False)
    st.write("Data (1) downloaded!")

    url = f"https://drive.google.com/uc?id={model_perfomance_id}"
    st.write("Downloading model from Google Drive...")
    gdown.download(url, model_perfomance_path, quiet=False)
    st.write("Data (1) downloaded!")

    data = joblib.load(data_path)
    df = pd.read_csv(model_perfomance_path)
    df.drop(columns=['Unnamed: 0'], inplace=True)
    
    return df, data
 
def get_recommendations(visitor_id, data, model, top_n=10):
    # Get a list of all unique items
    all_items = data['itemid'].unique()

    # Choose a specific user to make recommendations for
    visitor_id = visitor_id # Replace with a user ID from your dataset

    # Get the items the user has already interacted with
    user_interactions = data[data['visitorid'] == visitor_id]['itemid'].unique()

    # Get the items the user has not interacted with
    items_to_predict = [item for item in all_items if item not in user_interactions]

    # Predict ratings for the items the user has not interacted with
    predictions_for_user = []
    for item_id in items_to_predict:
        predicted_rating = model.predict(visitor_id, item_id).est
        predictions_for_user.append((item_id, predicted_rating))

    # Sort the predictions by predicted rating in descending order
    predictions_for_user.sort(key=lambda x: x[1], reverse=True)

    # Get the top N recommended items
    top_n_recommendations = predictions_for_user[:top_n] # Recommend top n items

    return top_n_recommendations

# --- Streamlit App ---
st.title("🎬 Movie Recommender System")

try:
    model = load_model()

    df, data = load_data()

    visitor_list = data[:20]['itemid'].tolist()
    suggested_number_of_recommendations = np.arange(10)[1:]

    st.dataframe(df)

    selected_visitor = st.selectbox("Choose a User:", visitor_list)
    number_of_recommendations = st.selectbox("How many Recommendations?", suggested_number_of_recommendations)

    if st.button("Get Recommendations"):
        with st.spinner("Generating recommendations..."):
            recommendations = get_recommendations(selected_visitor, data, model, number_of_recommendations)

            st.subheader("Recommended Movies:")
            ##for _, row in recommendations.iterrows():
            for item_id, predicted_rating in recommendations:
                st.write(f"**{item_id}**")
                st.text(f"Predicted Rating: {predicted_rating}")
                st.markdown("---")
except Exception as e:
    st.error(f"Error loading model or  {e}")
    st.write("Make sure all files (model, data(1), data(2)) are in the correct directory.")