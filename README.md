import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.pipeline import make_pipeline

# Load your training data (you might need to preprocess it accordingly)
train_data = pd.read_csv("/content/train_essays.csv")

# Split the data into training and validation sets
train_set, validation_set = train_test_split(train_data, test_size=0.2, random_state=42)

# Create a pipeline with a TF-IDF vectorizer and a random forest classifier
model = make_pipeline(
    TfidfVectorizer(max_features=5000),
    RandomForestClassifier(n_estimators=100, random_state=42)
)

# Train the model using the 'text' column as the feature and 'generated' column as the target
model.fit(train_set['text'], train_set['generated'])

# Make predictions on the validation set
validation_predictions = model.predict_proba(validation_set['text'])[:, 1]

# Evaluate the model using ROC AUC score on the validation set
roc_auc = roc_auc_score(validation_set['generated'], validation_predictions)
print(f"Validation ROC AUC: {roc_auc}")

# Predict the binary class (0 or 1) on the validation set
validation_predictions_binary = (validation_predictions > 0.5).astype(int)

# Calculate the accuracy on the validation set
accuracy = accuracy_score(validation_set['generated'], validation_predictions_binary)
print(f"Validation Accuracy: {accuracy}")

# Load your test data
test_data = pd.read_csv("/content/test_essays.csv")

# Make predictions on the test set
test_predictions = model.predict_proba(test_data['text'])[:, 1]

# Create a submission file
submission_df = pd.DataFrame({'id': test_data['id'], 'generated': test_predictions})
submission_df.to_csv("submission.csv", index=False)
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline

# Load the training data
train_data = pd.read_csv("/content/train_essays.csv")

# Split the data into training and validation sets
train_set, validation_set = train_test_split(train_data, test_size=0.2, random_state=42)

# Create a pipeline with a TF-IDF vectorizer and a random forest classifier
model = make_pipeline(
    TfidfVectorizer(max_features=5000),
    RandomForestClassifier(n_estimators=100, random_state=42)
)

# Train the model using the 'text' column as the feature and 'generated' column as the target
model.fit(train_set['text'], train_set['generated'])

# Make predictions on the validation set
validation_predictions = model.predict_proba(validation_set['text'])[:, 1]

# Evaluate the model using ROC AUC score on the validation set
roc_auc = roc_auc_score(validation_set['generated'], validation_predictions)
print(f"Validation ROC AUC: {roc_auc}")

# Load the test data
test_data = pd.read_csv("/content/test_essays.csv")

# Make predictions on the test set
test_predictions = model.predict_proba(test_data['text'])[:, 1]

# Create a submission file
submission_df = pd.DataFrame({'id': test_data['id'], 'generated': test_predictions})
submission_df.to_csv("submission.csv", index=False)
# Download the file
from google.colab import files
files.download("submission.csv")
