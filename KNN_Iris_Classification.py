#Example: Demonstrate KNN for a classification task using the Iris dataset:
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# Load the Iris dataset
iris = load_iris( )
X, y = iris.data, iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the feature data
scaler = StandardScaler( )
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)

# Fit the model on the training data
knn.fit(X_train, y_train)

# Predict the labels for the test set
y_pred = knn.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Print detailed classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

#display confusion matrix
mat = confusion_matrix(y_test, y_pred)
conf_disp = ConfusionMatrixDisplay(confusion_matrix=mat,display_labels=iris.target_names)
conf_disp.plot(cmap=plt.cm.Blues)
plt.title('Iris Confusion Matrix')
plt.show()

# Predict for a new sample
new_sample = np.array([[5.1, 3.5, 1.4, 0.2]])
new_sample = scaler.transform(new_sample)
prediction = knn.predict(new_sample)
print(f"Predicted class for the new sample: {iris.target_names[prediction][0]}")