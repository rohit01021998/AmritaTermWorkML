from libsvm import svmutil
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import numpy as np

# Load your dataset and prepare it in LIBSVM format
# Assuming you have your dataset in a CSV file named 'your_dataset.csv'
# Replace 'your_dataset.csv' with the actual filename if it's different.
df = pd.read_csv('generated_dataset\dataset_for_model.csv')

# Handling null values
df.dropna(inplace=True)

label_encoder = LabelEncoder()
df['Movement_Type_Encoded'] = label_encoder.fit_transform(df['Movement_Type'])

# Define features (X) and target (y)
X = df[['Local_X', 'Local_Y', 'v_Vel', 'v_Acc', 'Lateral_Velocity', 'Lateral_Acceleration', 'Yaw_Angle']]

# Scale the features using StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = df['Movement_Type_Encoded']

# Convert the data to LIBSVM's format
libsvm_data = svmutil.svm_problem(y.tolist(), X.tolist())

# Define SVM parameters (e.g., kernel, C value)
svm_parameters = svmutil.svm_parameter('-t 2 -c 10 -g 0.1')  # RBF kernel, C=1, and gamma=0.1

# Perform k-fold cross-validation (e.g., 5-fold cross-validation)
num_folds = 5
fold_size = len(X) // num_folds
accuracies = []

for i in range(num_folds):
    # Split the data into training and testing sets for this fold
    start_idx = i * fold_size
    end_idx = (i + 1) * fold_size

    X_test_fold = X[start_idx:end_idx]
    y_test_fold = y[start_idx:end_idx]

    X_train_fold = np.concatenate([X[:start_idx], X[end_idx:]], axis=0)
    y_train_fold = np.concatenate([y[:start_idx], y[end_idx:]], axis=0)

    # Convert the fold data to LIBSVM's format
    libsvm_train_data = svmutil.svm_problem(y_train_fold.tolist(), X_train_fold.tolist())
    libsvm_test_data = svmutil.svm_problem(y_test_fold.tolist(), X_test_fold.tolist())

    # Train the SVM classifier
    svm_model = svmutil.svm_train(libsvm_train_data, svm_parameters)

    # Make predictions on the test fold
    y_pred, accuracy, _ = svmutil.svm_predict(y_test_fold.tolist(), X_test_fold.tolist(), svm_model)

    # Store the accuracy for this fold
    accuracies.append(accuracy[0])

# Calculate the average accuracy across all folds
average_accuracy = sum(accuracies) / num_folds

# Print cross-validation results
print('############################')
print("Cross-Validation Results:")
print("Average Accuracy:", average_accuracy)
print('############################')
