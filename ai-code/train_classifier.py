import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

# Load the data
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Ensure consistent data shape (padding the sequences if needed)
data = data_dict['data']
labels = np.asarray(data_dict['labels'])

# Find the maximum length of the data sequences
max_len = max(len(item) for item in data)

# Pad sequences with zeros to ensure they all have the same length
padded_data = []
for item in data:
    padded_item = item + [0] * (max_len - len(item))  # Pad with zeros
    padded_data.append(padded_item)

# Convert to a numpy array (ensure it's a 2D array)
data = np.asarray(padded_data)

# Ensure that the data is in the correct format (2D array of shape (n_samples, n_features))
print(f"Data shape after padding: {data.shape}")

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Initialize the model
model = RandomForestClassifier()

# Train the model
model.fit(x_train, y_train)

# Make predictions
y_predict = model.predict(x_test)

# Calculate the accuracy score
score = accuracy_score(y_test, y_predict)

# Print the accuracy
print(f'Accuracy: {score * 100:.2f}%')

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_predict)
print(conf_matrix)

# Get F1-score
precision, recall, f1, support = precision_recall_fscore_support(y_test, y_predict, average='weighted')
print(f'F1 Score: {f1:.2f}')

# Save the trained model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
