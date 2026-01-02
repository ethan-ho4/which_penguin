from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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
    Calculates and prints accuracy, classification report, and confusion matrix.
    """
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    
    # 1. Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    
    # 2. Detailed Report (Precision, Recall, F1-Score)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # 3. Confusion Matrix (Where did it get confused?)
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return accuracy

def get_feature_importance(model, feature_names):
    """
    Returns a sorted series of feature importances.
    """
    importances = pd.Series(model.feature_importances_, index=feature_names)
    return importances.sort_values(ascending=False)
