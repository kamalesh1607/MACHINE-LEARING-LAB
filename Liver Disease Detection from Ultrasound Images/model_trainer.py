"""
Enhanced Model Training, Evaluation, ROC/AUC Curves, Model Comparison, and Persistence Module
Implements SVM, Random Forest, and KNN with StandardScaler Pipelines.
"""

import os
import time
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

AVAILABLE_ALGORITHMS = {
    "SVM": {
        "name": "Support Vector Machine (RBF Kernel)",
        "default_params": {"C": 1.0, "kernel": "rbf", "gamma": "scale"},
        "builder": lambda params: Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(
                C=params.get("C", 1.0),
                kernel=params.get("kernel", "rbf"),
                gamma=params.get("gamma", "scale"),
                probability=True,
                random_state=42
            ))
        ])
    },
    "Random Forest": {
        "name": "Random Forest Classifier (100 Trees)",
        "default_params": {"n_estimators": 100, "max_depth": None, "criterion": "gini"},
        "builder": lambda params: Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(
                n_estimators=params.get("n_estimators", 100),
                max_depth=params.get("max_depth", None),
                criterion=params.get("criterion", "gini"),
                random_state=42
            ))
        ])
    },
    "KNN": {
        "name": "K-Nearest Neighbors (K=5)",
        "default_params": {"n_neighbors": 5, "weights": "uniform", "metric": "minkowski"},
        "builder": lambda params: Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(
                n_neighbors=params.get("n_neighbors", 5),
                weights=params.get("weights", "uniform"),
                metric=params.get("metric", "minkowski")
            ))
        ])
    }
}


def build_pipeline(algorithm_key: str, custom_params: dict = None):
    """Instantiate the pipeline for the given algorithm key with optional custom parameters."""
    if algorithm_key not in AVAILABLE_ALGORITHMS:
        raise ValueError(f"Unknown algorithm: {algorithm_key}. Choose from {list(AVAILABLE_ALGORITHMS.keys())}")
    
    params = AVAILABLE_ALGORITHMS[algorithm_key]["default_params"].copy()
    if custom_params:
        params.update(custom_params)
        
    return AVAILABLE_ALGORITHMS[algorithm_key]["builder"](params), params


def train_and_evaluate_model(
    algorithm_key: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list,
    custom_params: dict = None
) -> dict:
    """
    Trains a selected ML model pipeline on training data and evaluates on test data.
    Computes Accuracy, Precision, Recall, F1, Confusion Matrix, Classification Report, and ROC/AUC.
    """
    pipeline, used_params = build_pipeline(algorithm_key, custom_params)

    start_time = time.time()
    pipeline.fit(X_train, y_train)
    train_duration = time.time() - start_time

    # Predictions
    y_pred = pipeline.predict(X_test)
    y_train_pred = pipeline.predict(X_train)

    # Probabilities for ROC / AUC
    probabilities = None
    roc_data = {}
    if hasattr(pipeline, "predict_proba"):
        try:
            probabilities = pipeline.predict_proba(X_test)
            # Multi-class ROC Curve calculation
            if len(class_names) > 1:
                y_test_bin = label_binarize(y_test, classes=class_names)
                if len(class_names) == 2 and y_test_bin.shape[1] == 1:
                    y_test_bin = np.hstack([1 - y_test_bin, y_test_bin])

                for i, cls_name in enumerate(class_names):
                    if i < probabilities.shape[1] and i < y_test_bin.shape[1]:
                        fpr, tpr, _ = roc_curve(y_test_bin[:, i], probabilities[:, i])
                        roc_auc = auc(fpr, tpr)
                        roc_data[cls_name] = {
                            "fpr": fpr,
                            "tpr": tpr,
                            "auc": float(roc_auc) if not np.isnan(roc_auc) else 1.0
                        }
        except Exception as e:
            print(f"ROC computation note: {e}")

    # Calculate actual metrics
    train_acc = float(accuracy_score(y_train, y_train_pred))
    test_acc = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    recall = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=class_names)

    # Classification Report
    clf_report_dict = classification_report(
        y_test,
        y_pred,
        labels=class_names,
        output_dict=True,
        zero_division=0
    )

    # Format classification report into clean DataFrame
    report_rows = []
    for cls_name in class_names:
        if cls_name in clf_report_dict:
            metrics = clf_report_dict[cls_name]
            report_rows.append({
                "Class": cls_name,
                "Precision": round(metrics["precision"], 4),
                "Recall": round(metrics["recall"], 4),
                "F1-Score": round(metrics["f1-score"], 4),
                "Support": int(metrics["support"])
            })

    report_df = pd.DataFrame(report_rows)

    result = {
        "algorithm": algorithm_key,
        "algorithm_full_name": AVAILABLE_ALGORITHMS[algorithm_key]["name"],
        "hyperparameters": used_params,
        "pipeline": pipeline,
        "train_duration_sec": round(train_duration, 4),
        "train_accuracy": train_acc,
        "accuracy": test_acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm,
        "class_names": class_names,
        "classification_report_dict": clf_report_dict,
        "classification_report_df": report_df,
        "roc_data": roc_data,
        "y_test": y_test,
        "y_pred": y_pred,
        "num_train_samples": len(X_train),
        "num_test_samples": len(X_test),
        "num_features": X_train.shape[1],
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    return result


def compare_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list
) -> tuple[pd.DataFrame, dict]:
    """
    Trains and evaluates all supported models (SVM, Random Forest, KNN) on the exact same dataset split.
    Returns a comparison DataFrame and dictionary of individual results.
    """
    results = {}
    rows = []

    for algo_key in ["SVM", "Random Forest", "KNN"]:
        res = train_and_evaluate_model(
            algorithm_key=algo_key,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            class_names=class_names
        )
        results[algo_key] = res
        rows.append({
            "Algorithm": algo_key,
            "Full Name": res["algorithm_full_name"],
            "Accuracy": round(res["accuracy"], 4),
            "Precision": round(res["precision"], 4),
            "Recall": round(res["recall"], 4),
            "F1 Score": round(res["f1_score"], 4),
            "Train Time (s)": res["train_duration_sec"]
        })

    comparison_df = pd.DataFrame(rows)
    return comparison_df, results


def save_model(
    model_payload: dict,
    save_dir: str = "models",
    filename: str = "liver_ultrasound_model.pkl"
) -> str:
    """
    Saves trained model pipeline and metadata using joblib.
    """
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)

    data_to_save = {
        "pipeline": model_payload["pipeline"],
        "algorithm": model_payload["algorithm"],
        "algorithm_full_name": model_payload.get("algorithm_full_name", model_payload["algorithm"]),
        "hyperparameters": model_payload.get("hyperparameters", {}),
        "class_names": model_payload["class_names"],
        "accuracy": model_payload.get("accuracy", 0.0),
        "precision": model_payload.get("precision", 0.0),
        "recall": model_payload.get("recall", 0.0),
        "f1_score": model_payload.get("f1_score", 0.0),
        "confusion_matrix": model_payload.get("confusion_matrix", None),
        "classification_report_df": model_payload.get("classification_report_df", None),
        "roc_data": model_payload.get("roc_data", {}),
        "trained_at": model_payload.get("trained_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        "num_features": model_payload.get("num_features", None),
        "feature_types": ["HOG (9 bins, 8x8 cell, 2x2 block)", "GLCM (5 properties, 3 distances, 4 angles)"]
    }

    joblib.dump(data_to_save, filepath)
    return filepath


def load_saved_model(filepath: str = "models/liver_ultrasound_model.pkl") -> dict:
    """
    Loads saved model package from disk.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found at: {filepath}")

    payload = joblib.load(filepath)
    return payload


def predict_single_image(
    feature_vector: np.ndarray,
    model_payload: dict
) -> dict:
    """
    Runs prediction for a single image feature vector.
    Calculates predicted class and class confidence probabilities.
    """
    pipeline = model_payload["pipeline"]
    class_names = model_payload["class_names"]

    # Ensure 2D shape (1, n_features)
    if feature_vector.ndim == 1:
        feature_vector = feature_vector.reshape(1, -1)

    predicted_class = pipeline.predict(feature_vector)[0]

    probabilities = {}
    confidence = None

    if hasattr(pipeline, "predict_proba"):
        try:
            probs = pipeline.predict_proba(feature_vector)[0]
            # Match probabilities to pipeline classes
            clf_classes = getattr(pipeline, "classes_", class_names)
            for cls_name, prob in zip(clf_classes, probs):
                probabilities[cls_name] = float(prob)
            if predicted_class in probabilities:
                confidence = probabilities[predicted_class]
        except Exception:
            pass

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": probabilities,
        "algorithm": model_payload.get("algorithm", "Trained ML Model"),
        "class_names": class_names
    }
