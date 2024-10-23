import joblib
from flask import Flask, render_template, request, jsonify
import sklearn
print(sklearn.__version__)

app = Flask(__name__)

# Load the Random Forest model from the saved file
# random_forest_model = joblib.load('fetal_health1.joblib')
import os
import pickle

model_path = os.path.abspath("fetal_health12.pkl")

# Save the model
with open(model_path, 'wb') as file:
    pickle.dump(model_path, file)

# Load the model
with open(model_path, 'rb') as file:
    loaded_model = pickle.load(file)


# print("Number of Estimators:", len(random_forest_model.estimators_))
# print("Feature Importances:", random_forest_model.feature_importances_)

@app.route('/')
def index5():
    return render_template('testing/index5.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = [data['accelerations'], data['prolongued_decelerations'], data['abnormal_short_term_variability'], data['percentage_of_time_with_abnormal_long_term_variability'],
                data['mean_value_of_long_term_variability'], data['histogram_mode'], data['histogram_median'], data['histogram_variance']]
    prediction = model_path.predict([features])[0]
    return jsonify(prediction=str(prediction))
    
if __name__ == '__main__':
    app.run(debug=True)
