# Heart Failure Classification using SVM with Firefly Algorithm Optimization

## Project Description
This project implements a machine learning model to predict heart failure outcomes using Support Vector Machines (SVM) optimized with Firefly Algorithm. The model addresses class imbalance through SMOTE (Synthetic Minority Over-sampling Technique) and includes comprehensive data preprocessing.

## Features
- Data preprocessing with outlier detection and winsorization
- Feature scaling using StandardScaler
- Class imbalance handling with SMOTE
- SVM hyperparameter optimization (C and gamma) using Firefly Algorithm
- Model evaluation with accuracy metrics and confusion matrix

## Installation
1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dependencies
- Python 3.x
- numpy
- pandas
- seaborn
- matplotlib
- scikit-learn
- imbalanced-learn

## Dataset
The project uses the "heart_failure_clinical_records_dataset.csv" dataset containing clinical records of heart failure patients with the following features:
- Age
- Anaemia
- Creatinine phosphokinase
- Diabetes
- Ejection fraction
- High blood pressure
- Platelets
- Serum creatinine
- Serum sodium
- Sex
- Smoking
- Time
- DEATH_EVENT (target variable)

## Usage
1. Run the main script:
```bash
python main.py
```

The script will:
- Load and preprocess the data
- Handle class imbalance
- Optimize SVM parameters using Firefly Algorithm
- Train and evaluate the model
- Save the trained model and scaler as pickle files

## Methodology
1. **Data Preprocessing**:
   - Outlier detection using IQR method
   - Winsorization to handle outliers
   - Feature scaling with StandardScaler

2. **Class Imbalance Handling**:
   - SMOTE oversampling to balance classes

3. **Model Training**:
   - SVM with RBF kernel
   - Firefly Algorithm optimization for hyperparameters (C and gamma)

4. **Evaluation**:
   - 10-fold cross-validation during optimization
   - Final evaluation on test set with accuracy, precision, recall, and F1-score
   - Confusion matrix visualization

## Results
The optimized SVM model achieves:
- Accuracy: [final accuracy from output]
- Precision, Recall, and F1-score for each class (see classification report)

## License
[MIT License](LICENSE)