from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

def train_model(X_train, y_train):
    """
    Trains a Random Forest Classifier.
    """
    print("Training Random Forest model...")
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    return rf

def evaluate_model(model, X_test, y_test):
    """
    Calculates and prints accuracy.
    """
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")
    return accuracy

def get_feature_importance(model, feature_names):
    """
    Returns a sorted series of feature importances.
    """
    importances = pd.Series(model.feature_importances_, index=feature_names)
    return importances.sort_values(ascending=False)
