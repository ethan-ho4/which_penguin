import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_clean_data():
    """
    Loads the penguins dataset and drops rows with missing values.
    Returns:
        df (pd.DataFrame): Cleaned dataframe.
    """
    print("Loading and cleaning data...")
    df = sns.load_dataset('penguins')
    df.dropna(inplace=True)
    return df

def preprocess_data(df):
    """
    Encodes categorical variables and splits the data.
    Args:
        df (pd.DataFrame): Cleaned dataframe.
    Returns:
        X_train, X_test, y_train, y_test (tuple): Split data.
        feature_names (list): List of feature names after encoding.
    """
    print("Preprocessing data...")
    # One-hot encode categorical variables
    df = pd.get_dummies(df, columns=['island', 'sex'], drop_first=True)
    
    # Define features and target
    X = df.drop('species', axis=1)
    y = df['species']
    
    feature_names = X.columns
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, feature_names