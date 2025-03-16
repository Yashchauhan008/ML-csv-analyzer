# CSV Analyzer - Machine Learning Model Evaluator

## Overview
The CSV Analyzer is a simple yet powerful tool that takes a CSV file as input and provides an analysis of its dataset. It evaluates multiple machine learning models on the dataset and recommends the best-performing model based on accuracy. Additionally, it allows users to make predictions based on input features.

## Features
- 📦 **Dataset Information**: Displays the number of rows and columns in the provided CSV file.
- ✅ **Model Recommendation**: Suggests the best-performing model with the highest accuracy.
- 🏅 **Model Performance Metrics**: Evaluates multiple machine learning models and presents accuracy, precision, recall, and F1 score.
- 🔮 **Prediction Functionality**: Users can input specific values and make predictions using the trained models.
- ⬅️ **File Upload Support**: Users can upload different CSV files for analysis.

## Supported Machine Learning Models
The following models are evaluated and compared:
- **Logistic Regression**
- **Decision Tree**
- **Random Forest**
- **Support Vector Machine (SVM)**
- **K-Nearest Neighbors (KNN)**
- **Naive Bayes**
- **AdaBoost**

## How It Works
1. **Upload CSV File**: The user uploads a CSV dataset.
2. **Data Analysis**: The tool extracts the number of rows and columns.
3. **Model Training & Evaluation**:
   - Each supported model is trained on the dataset.
   - Accuracy, precision, recall, and F1 score are calculated for each model.
   - The best-performing model is recommended.
4. **Prediction**:
   - The user can input feature values.
   - The recommended model predicts the outcome.

## Example Output
```
📊 Machine Learning Analysis Results
📦 Dataset Information
Rows: 1025
Columns: 14

✅ Recommended Model: Decision Tree with Accuracy: 99.71%

🏅 Model Performance
Model		Accuracy	Precision	Recall	F1 Score
Logistic Regression	84.59%	84.94%	84.59%	84.53%
Decision Tree	99.71%	99.72%	99.71%	99.71%
Random Forest	99.71%	99.72%	99.71%	99.71%
Support Vector Machine	91.61%	91.74%	91.61%	91.6%
K-Nearest Neighbors	84.98%	85.32%	84.98%	84.97%
Naive Bayes	82.05%	82.38%	82.05%	82.0%
AdaBoost	90.93%	90.95%	90.93%	90.93%
```

## Installation & Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/Yashchauhan008/ML-csv-analyzer.git
   cd ML-csv-analyzer
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   python app.py
   ```
4. Upload a CSV file and analyze its performance.

## Requirements
- Python 3.x
- Pandas
- Scikit-learn
- Flask (if using a web interface)
