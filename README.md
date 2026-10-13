# Wine Quality Prediction (UCI Wine Quality Dataset)

Mini project for U23AMT01 - Data Science. Predicts wine quality from chemical
properties using **regression** (quality score) and **classification** (Good = quality >= 6).

## Run
```
pip install -r requirements.txt
python wine_quality.py      # downloads dataset (or place CSVs in data/), trains models, saves outputs/
python build_report.py      # builds Wine_Quality_Prediction_Report.pdf from outputs/ (edit your name/reg no. first)
```
Dataset: https://archive.ics.uci.edu/dataset/186/wine+quality
(`winequality-red.csv`, `winequality-white.csv` -> `data/`)

## Models
- Regression: Linear, Ridge, Random Forest, Gradient Boosting (MAE, RMSE, R2)
- Classification: Logistic Regression, KNN, Decision Tree, SVM, Random Forest (Accuracy, Precision, Recall, F1, 5-fold CV)
