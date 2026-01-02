from data_utils import load_and_clean_data, preprocess_data
from model import train_model, evaluate_model, get_feature_importance

def main():
    # 1. Loading & Cleaning
    df = load_and_clean_data()
    print(f"Data loaded. Shape: {df.shape}")

    # 2. Pre-processing
    X_train, X_test, y_train, y_test, feature_names = preprocess_data(df)
    print(f"Training set: {X_train.shape}, Testing set: {X_test.shape}")

    # 3. Modeling
    model = train_model(X_train, y_train)

    # 4. Evaluation
    evaluate_model(model, X_test, y_test)
    
    # Feature Importance
    print("\nFeature Importances:")
    print(get_feature_importance(model, feature_names))

if __name__ == "__main__":
    main()
