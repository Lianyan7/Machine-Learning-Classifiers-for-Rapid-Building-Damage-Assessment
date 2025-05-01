import os
import time
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    f1_score, recall_score, precision_score
)


def load_and_preprocess(file_path):
    """
    Load data from Excel sheets and preprocess features and labels.
    """
    # Read sheets
    df_value = pd.read_excel(file_path, sheet_name='Value')
    df_multi = pd.read_excel(file_path, sheet_name='Multi-class')
    df_binary = pd.read_excel(file_path, sheet_name='Binary')
    df_label = pd.read_excel(file_path, sheet_name='Label')

    # Features and label
    X_value = df_value.values
    X_multi = pd.get_dummies(df_multi)
    X_binary = pd.get_dummies(df_binary)
    y = df_label.values.ravel()

    # Normalize
    scaler = StandardScaler()
    X_value_norm = scaler.fit_transform(X_value)

    # Combine all
    X = np.concatenate([X_value_norm, X_multi, X_binary], axis=1)
    return X, y


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Train model, predict and return metrics and confusion matrix.
    """
    # Train
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    cm = confusion_matrix(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    accuracy = accuracy_score(y_test, y_pred)

    # Training set metrics
    y_train_pred = model.predict(X_train)
    train_f1 = f1_score(y_train, y_train_pred, average='weighted')
    train_precision = precision_score(y_train, y_train_pred, average='weighted')
    train_recall = recall_score(y_train, y_train_pred, average='weighted')
    train_accuracy = accuracy_score(y_train, y_train_pred)

    metrics = {
        'confusion_matrix': cm,
        'f1_weighted': f1,
        'recall_weighted': recall,
        'precision_weighted': precision,
        'accuracy': accuracy,
        'train_f1': train_f1,
        'train_precision': train_precision,
        'train_recall': train_recall,
        'train_accuracy': train_accuracy,
        'training_time_s': train_time
    }
    return metrics


def grid_search_gb(X_train, y_train, param_grid, cv=5, n_jobs=-1):
    gb = GradientBoostingClassifier(random_state=42)
    gs = GridSearchCV(
        estimator=gb,
        param_grid=param_grid,
        scoring='f1_weighted',
        cv=cv,
        n_jobs=n_jobs
    )
    gs.fit(X_train, y_train)
    return gs.best_estimator_, gs.best_params_


def grid_search_gbag(X_train, y_train, param_grid, cv=5, n_jobs=-1):
    base = GradientBoostingClassifier(random_state=42)
    bag = BaggingClassifier(estimator=base, random_state=42)
    gs = GridSearchCV(
        estimator=bag,
        param_grid=param_grid,
        scoring='f1_weighted',
        cv=cv,
        n_jobs=n_jobs
    )
    gs.fit(X_train, y_train)
    return gs.best_estimator_, gs.best_params_


def save_results(results, prefix, params):
    """
    Save metrics to Excel file.
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_results_{timestamp}.xlsx"
    df = pd.DataFrame({
        'Metric': list(results.keys()),
        'Value': list(results.values())
    })
    df.to_excel(filename, index=False)
    print(f"Results saved to: {filename}")


def main(args):
    X, y = load_and_preprocess(args.input)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42
    )

    # Gradient Boosting
    gb_params = {
        'n_estimators': args.gb_n_estimators,
        'learning_rate': args.gb_learning_rate,
        'max_depth': args.gb_max_depth
    }
    gb_param_grid = {
        'n_estimators': [int(x) for x in gb_params['n_estimators'].split(',')],
        'learning_rate': [float(x) for x in gb_params['learning_rate'].split(',')],
        'max_depth': [int(x) for x in gb_params['max_depth'].split(',')]
    }
    best_gb, best_gb_params = grid_search_gb(X_train, y_train, gb_param_grid)
    print("GB Optimal Hyperparameters:", best_gb_params)
    gb_metrics = evaluate_model(best_gb, X_train, X_test, y_train, y_test)
    save_results(gb_metrics, 'GB', best_gb_params)

    # Gradient Bagging
    gbag_param_grid = {
        'estimator__n_estimators': gb_param_grid['n_estimators'],
        'estimator__learning_rate': gb_param_grid['learning_rate'],
        'estimator__max_depth': gb_param_grid['max_depth'],
        'n_estimators': [10, 20, 30]
    }
    best_gbag, best_gbag_params = grid_search_gbag(X_train, y_train, gbag_param_grid)
    print("GBag Optimal Hyperparameters:", best_gbag_params)
    gbag_metrics = evaluate_model(best_gbag, X_train, X_test, y_train, y_test)
    save_results(gbag_metrics, 'GBag', best_gbag_params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compare GradientBoosting and Gradient Bagging on RBA data.'
    )
    parser.add_argument('--input', type=str, default='RBA with three labels.xlsx',
                        help='Path to Excel file with RBA data')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of data for testing')
    parser.add_argument('--gb_n_estimators', type=str, default='50,100,150',
                        help='Comma-separated GB n_estimators grid')
    parser.add_argument('--gb_learning_rate', type=str, default='0.01,0.1,0.2',
                        help='Comma-separated GB learning_rate grid')
    parser.add_argument('--gb_max_depth', type=str, default='3,4,5',
                        help='Comma-separated GB max_depth grid')
    args = parser.parse_args()
    main(args)
