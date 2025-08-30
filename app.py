from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
from surprise import SVD
import os

app = Flask(__name__)

# Define file paths
DATA_PATH = 'data/data.pkl'
MODEL_PATH = 'model/model.pkl'
PERFORMANCE_PATH = 'data/comparison_df_set_1.csv'

# Load model ONCE at startup
try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Error loading model:", str(e))
    model = None

# Load data ONCE at startup
try:
    data = joblib.load(DATA_PATH)
    print("✅ Data loaded successfully")
except Exception as e:
    print("❌ Error loading data:", str(e))
    data = None

# Load model performance table
try:
    df = pd.read_csv(PERFORMANCE_PATH)
    # Convert to list of dicts for easy rendering
    MODEL_PERFORMANCE = df.to_dict(orient='records')
    print("✅ Model Performance data loaded from CSV")
except Exception as e:
    print("❌ Error loading Model performance data CSV:", str(e))
    MODEL_PERFORMANCE = []  # Fallback


def get_recommendations(visitor_id, data, model, top_n=10):
    if visitor_id not in data['visitorid'].values:
        return []  # Return empty if user not found

    all_items = data['itemid'].unique()
    user_interactions = data[data['visitorid'] == visitor_id]['itemid'].unique()
    items_to_predict = [item for item in all_items if item not in user_interactions]

    predictions_for_user = []
    for item_id in items_to_predict:
        try:
            predicted_rating = model.predict(visitor_id, item_id).est
            predictions_for_user.append((item_id, predicted_rating))
        except Exception:
            continue  # Skip failed predictions

    predictions_for_user.sort(key=lambda x: x[1], reverse=True)
    return predictions_for_user[:top_n]


@app.route('/')
def index():
    return render_template('index.html', models=MODEL_PERFORMANCE)


@app.route('/recommend', methods=['POST'])
def recommend():
    if data is None or model is None:
        return jsonify({"error": "Model or data failed to load"}), 500

    try:
        visitor_id = int(request.form.get('user_id'))
        n = int(request.form.get('n_recs'))

        if visitor_id not in data['visitorid'].values:
            return jsonify({"error": f"User {visitor_id} not found in the dataset."}), 400

        recs = get_recommendations(visitor_id, data, model, top_n=n)
        result = [
            {"movie_id": int(item_id), "predicted_rating": float(rating)}
            for item_id, rating in recs
        ]
        return jsonify(result)

    except ValueError:
        return jsonify({"error": "Invalid input: user ID and number must be integers."}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)