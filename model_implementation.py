from libsvm import svmutil
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt

# Load your dataset and prepare it in LIBSVM format
# Make sure your dataset is in the format accepted by LIBSVM

# Assuming you have your dataset in a CSV file named 'your_dataset.csv'
# Replace 'your_dataset.csv' with the actual filename if it's different.
import pandas as pd
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


# Split the data into a training set and a testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the data to LIBSVM's format
libsvm_train_data = svmutil.svm_problem(y_train.tolist(), X_train.tolist())
libsvm_test_data = svmutil.svm_problem(y_test.tolist(), X_test.tolist())

# Define SVM parameters (e.g., kernel, C value)
svm_parameters = svmutil.svm_parameter('-t 2 -c 10 -g 0.1')  # RBF kernel, C=1, and gamma=0.1

# Train the SVM classifier
svm_model = svmutil.svm_train(libsvm_train_data, svm_parameters)

# Make predictions on the test set
y_pred, accuracy, _ = svmutil.svm_predict(y_test.tolist(), X_test.tolist(), svm_model)

# Print accuracy and other evaluation metrics
print("Accuracy:", accuracy)

cm = confusion_matrix(y_test, y_pred)


cm = confusion_matrix(y_test, y_pred)

# Display the confusion matrix
display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
display.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix")
plt.show()





