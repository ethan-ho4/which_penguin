# Penguin Species Classifier 🐧

A Machine Learning project that correctly identifies penguin species (**Adélie**, **Chinstrap**, **Gentoo**) based on their physical measurements.

## 📌 Overview
This project uses the **Random Forest** algorithm to predict a penguin's species with **100% accuracy** on the test set. It demonstrates key Data Science concepts including data cleaning, one-hot encoding, model training, and feature importance analysis.

## 📊 Dataset
We use the **Palmer Penguins** dataset (via `seaborn`), which includes:
- **Species**: Adélie, Chinstrap, Gentoo (Target)
- **Measurements**: Bill Length, Bill Depth, Flipper Length, Body Mass
- **Metadata**: Island, Sex

## 🏗️ Project Structure
The project is modularized for scalability:
- **`main.py`**: The entry point. Orchestrates data loading, training, and evaluation.
- **`data_utils.py`**: Handles loading, cleaning (dropping NaNs), and preprocessing (encoding & splitting).
- **`model.py`**: Contains the Random Forest implementation and detailed evaluation metrics (Accuracy, Confusion Matrix, Classification Report).
- **`show_data.py`**: A helper script to quickly visualize the raw dataset.

## 🚀 Getting Started

### Prerequisites
You need Python installed along with the following libraries:
```bash
pip install pandas seaborn scikit-learn
```

### Usage
Run the main script to train the model and see the results:
```bash
python main.py
```

### Example Output
```text
Test Accuracy: 1.0000

Feature Importances:
1. bill_length_mm
2. flipper_length_mm
3. bill_depth_mm
...
```

## 🧠 Key Findings
- **Bill Length** and **Flipper Length** are the most critical features for distinguishing these three species.
- The model achieves perfect classification on the held-out test set (20% of data).
