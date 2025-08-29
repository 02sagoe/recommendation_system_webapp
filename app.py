import streamlit as st
import pandas as pd
import numpy as np
import joblib
from surprise import SVD

# Function to download and load the model 
@st.cache_resource
def load_model():
    model = joblib.load('model/model.pkl')
    
    return model

# Function to download and load the datasets
def load_data():
    data = joblib.load('data/data.pkl')

    df = pd.read_csv('data/comparison_df_set_1.csv')
        
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