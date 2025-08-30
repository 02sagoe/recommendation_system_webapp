# app.py
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Example model performance data
MODEL_PERFORMANCE = [
    {"Model": "SVD++", "RMSE": 0.222, "Time (seconds)": 423.668},
    {"Model": "SVD++ - cached ratings", "RMSE": 0.222, "Time (seconds)": 285.96},
    {"Model": "SVD n_factors - 50", "RMSE": 0.223, "Time (seconds)": 40.725},
    {"Model": "SVD n_factors - 100", "RMSE": 0.2252, "Time (seconds)": 55.433},
    {"Model": "CoClustering", "RMSE": 0.264, "Time (seconds)": 163.994}
]

def get_recommendations(user_id, n):
    # Replace with your actual ML logic
    recommendations = [
        {"movie_id": 132633, "predicted_rating": 1.7518331778904646},
        {"movie_id": 28789, "predicted_rating": 1.5829283270816574}
    ]
    return recommendations[:n]

@app.route('/')
def index():
    return render_template('index.html', models=MODEL_PERFORMANCE)

@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = request.form.get('user_id')
    n = int(request.form.get('n_recs'))
    recs = get_recommendations(user_id, n)
    return jsonify(recs)

if __name__ == '__main__':
    app.run(debug=True, port=5000)