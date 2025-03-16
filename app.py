import os
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, jsonify
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_data(df):
    df.fillna(df.mean(numeric_only=True), inplace=True)
    df.fillna('', inplace=True)

    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
        label_encoders[column] = le

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, df, scaler

def evaluate_model(model, X, y, is_regression=False):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    acc_scores = []
    prec_scores = []
    rec_scores = []
    f1_scores = []
    r2_scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if is_regression:
            r2_scores.append(r2_score(y_test, y_pred))
        else:
            acc_scores.append(accuracy_score(y_test, y_pred))
            prec_scores.append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
            rec_scores.append(recall_score(y_test, y_pred, average='weighted', zero_division=0))
            f1_scores.append(f1_score(y_test, y_pred, average='weighted'))

    if is_regression:
        return {'R² Score': round(np.mean(r2_scores) * 100, 2)}
    else:
        return {
            'accuracy': round(np.mean(acc_scores) * 100, 2),
            'precision': round(np.mean(prec_scores) * 100, 2),
            'recall': round(np.mean(rec_scores) * 100, 2),
            'f1_score': round(np.mean(f1_scores) * 100, 2)
        }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():

    global feature_names, best_model, scaler, processed_df  # Ensure global scope

    if 'file' not in request.files:
        return "No file uploaded!"

    file = request.files['file']

    if file.filename == '':
        return "No file selected!"

    if file and allowed_file(file.filename):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        df = pd.read_csv(filepath)
        X, y, processed_df, scaler = preprocess_data(df)  # Save processed_df globally
        feature_names = processed_df.columns[:-1]

        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(),
            'Random Forest': RandomForestClassifier(),
            'Support Vector Machine': SVC(),
            'K-Nearest Neighbors': KNeighborsClassifier(),
            'Naive Bayes': GaussianNB(),
            'AdaBoost': AdaBoostClassifier()
        }

        results = []
        best_accuracy = 0
        best_model_name = None

        for model_name, model in models.items():
            is_regression = model_name == "Linear Regression"
            metrics = evaluate_model(model, X, y, is_regression)
            results.append({'model': model_name, **metrics})

            if 'accuracy' in metrics and metrics['accuracy'] > best_accuracy:
                best_accuracy = metrics['accuracy']
                best_model_name = model_name

        # Train best model
        best_model = models[best_model_name]
        best_model.fit(X, y)

        dataset_info = {
            'shape': processed_df.shape,
            'missing_values': processed_df.isnull().sum().to_dict(),
            'data_types': processed_df.dtypes.apply(str).to_dict()
        }

        return render_template('results.html',
                               results=results,
                               feature_names=feature_names,
                               dataset_info=dataset_info)
    else:
        return "Invalid file type. Only CSV files are allowed!"
@app.route('/predict', methods=['POST'])
def predict():
    try:
        global best_model, feature_names, scaler, processed_df  # Ensure access to global variables

        if 'best_model' not in globals() or best_model is None:
            return jsonify({"error": "Model is not trained! Please upload a dataset first."})

        data = request.json
        
        missing_features = [feature for feature in feature_names if feature not in data]
        if missing_features:
            return jsonify({"error": f"Missing features: {missing_features}"})

        # Convert JSON input into NumPy array and scale it
        input_values = np.array([float(data[feature]) for feature in feature_names]).reshape(1, -1)
        input_scaled = scaler.transform(input_values)

        # Make prediction
        prediction = best_model.predict(input_scaled)[0]

        return jsonify({"prediction": int(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)