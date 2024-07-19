import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

def load_data(file_path):
    df = pd.read_csv(file_path)
    X = df.drop(columns=['anxiety_meter', 'P_Id'], axis=1)
    y = df['anxiety_meter']
    y = (y >= 16).astype(int)
    y.name = 'IsAnxious'
    return X, y

def create_pipeline(classifier):
    return Pipeline([
        ('imputer', KNNImputer(n_neighbors=5)),  # Replace missing values with KNNImputer
        ('scaler', RobustScaler()),  # Scale features using RobustScaler
        ('kbest', SelectKBest(score_func=f_classif, k=15)),  # Select features using SelectKBest
        ('classifier', classifier),  # Classifier
    ])

def get_param_grids():
    return {
        'svc': {
            'classifier__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
            'classifier__C': [0.1, 1.0, 10.0, 100.0],
            'classifier__gamma': ['scale', 'auto', 0.01, 0.001, 0.0001]
        },
        'rfc': {
            'classifier': [RandomForestClassifier(random_state=42)],
            'classifier__n_estimators': [100, 200, 300, 500],
            'classifier__max_depth': [None, 10, 20, 30, 50],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4],
            'classifier__max_features': ['sqrt', 'log2']
        },
        'logreg': {
            'classifier': [LogisticRegression(random_state=42)],
            'classifier__solver': ['lbfgs', 'liblinear', 'saga'],
            'classifier__C': [0.1, 1.0, 10.0, 100.0],
            'classifier__max_iter': [100, 200, 300]
        },
        'xgboost': {
            'classifier': [XGBClassifier(random_state=42)],
            'classifier__learning_rate': [0.1, 0.01, 0.001],
            'classifier__n_estimators': [100, 200, 300, 500],
            'classifier__max_depth': [3, 5, 7, 10],
            'classifier__subsample': [0.5, 0.7, 1.0],
            'classifier__colsample_bytree': [0.5, 0.7, 1.0]
        },
        'dtc': {
            'classifier': [DecisionTreeClassifier(random_state=42)],
            'classifier__max_depth': [None, 10, 20, 30, 50],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4],
            'classifier__max_features': ['sqrt', 'log2']
        },
        'nbayes': {
            'classifier': [GaussianNB()]
        },
        'mlp': {
            'classifier': [MLPClassifier(random_state=42, max_iter=500)],
            'classifier__hidden_layer_sizes': [(50,), (100,), (100, 50)],
            'classifier__activation': ['tanh', 'relu'],
            'classifier__solver': ['adam', 'sgd'],
            'classifier__alpha': [0.0001, 0.001, 0.01]
        },
        'voting-algorithm': {
            'classifier': [VotingClassifier(estimators=[
                ('svc', SVC(probability=True, random_state=42)),
                ('rfc', RandomForestClassifier(random_state=42)),
                ('logreg', LogisticRegression(random_state=42)),
                ('xgboost', XGBClassifier(random_state=42)),
                ('dtc', DecisionTreeClassifier(random_state=42))
            ], voting='soft')]
        }
    }

def perform_grid_search(X, y, param_grids):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    for model_name, param_grid in param_grids.items():
        if 'classifier' in param_grid:
            classifier = param_grid['classifier'][0]
            del param_grid['classifier']
            pipeline = create_pipeline(classifier)
        else:
            pipeline = create_pipeline(SVC(random_state=42))

        grid_search = GridSearchCV(pipeline, param_grid, cv=skf, scoring='accuracy', verbose=1, n_jobs=-1)
        grid_search.fit(X, y)
        results[model_name] = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'best_std_deviation': grid_search.cv_results_['std_test_score'][grid_search.best_index_]
        }
    return results

def print_results(results):
    for model_name, result in results.items():
        print(f"Model: {model_name}")
        print("Best parameters:", result['best_params'])
        print("Best cross-validation score:", result['best_score'])
        print("Best standard deviation:", result['best_std_deviation'])
        print("-" * 50)

def generate_pdf_report(results, filename='model_comparison_report.pdf'):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Model Comparison Report', 0, 1, 'C')
    pdf.ln(10)
    
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Summary of Results', 0, 1, 'L')
    pdf.ln(5)
    
    pdf.set_font('Arial', '', 12)
    
    for model_name, result in results.items():
        pdf.cell(0, 10, f"Model: {model_name}", 0, 1, 'L')
        pdf.cell(0, 10, f"Best Parameters: {result['best_params']}", 0, 1, 'L')
        pdf.cell(0, 10, f"Best Cross-Validation Score: {result['best_score']:.4f}", 0, 1, 'L')
        pdf.cell(0, 10, f"Best Standard Deviation: {result['best_std_deviation']:.4f}", 0, 1, 'L')
        pdf.ln(10)
    
    pdf.output(filename)

if __name__ == "__main__":
    X, y = load_data('Datasets/reduced_II.csv')
    param_grids = get_param_grids()
    results = perform_grid_search(X, y, param_grids)
    
    # Print results to console
    print_results(results)
    
    # Generate PDF report
    generate_pdf_report(results, 'model_comparison_report.pdf')
