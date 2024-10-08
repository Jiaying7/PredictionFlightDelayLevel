# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

# Load the dataset
path_to_your_dataset = '/Users/yujiaying/Desktop/Python/CA683/flight_total.csv'
df = pd.read_csv(path_to_your_dataset)
# Convert FL_DATE to datetime and extract more features
df['FL_DATE'] = pd.to_datetime(df['FL_DATE'], format='%d/%m/%Y')
df['YEAR'] = df['FL_DATE'].dt.year
df['MONTH'] = df['FL_DATE'].dt.month
df['DAY'] = df['FL_DATE'].dt.day
# drop the original FL_DATE column
df.drop(['FL_DATE'], axis=1, inplace=True)
# Prepare the feature matrix (X) and the target vector (y)
X = df[['YEAR', 'MONTH', 'DAY', 'OP_CARRIER', 'ORIGIN_CITY_NAME', 'CRS_DEP_TIME']]
y = df['FLIGHT_DELAY_LEVEL']
# Define a preprocessor for the pipeline that scales numerical features and encodes categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['YEAR', 'MONTH', 'DAY', 'CRS_DEP_TIME']),
        ('cat', OneHotEncoder(), ['OP_CARRIER', 'ORIGIN_CITY_NAME'])
    ])
# Create a pipeline that preprocesses the data and then applies KNN
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', KNeighborsClassifier(n_neighbors=5))
])
# Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train the model
pipeline.fit(X_train, y_train)
# Predict on the test set
y_pred = pipeline.predict(X_test)
# Predict probabilities for the test set
y_proba = pipeline.predict_proba(X_test)
# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
# Calculate and print the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)
# Calculate and print the classification report with zero_division parameter set to 0
class_report = classification_report(y_test, y_pred, target_names=[f'Group {i}' for i in range(6)], zero_division=0)
print("\nClassification Report:")
print(class_report)
# Calculate probabilities for each group
probabilities = np.mean(y_proba, axis=0)
print("\nPredicted Probabilities for Each Group:")
for i, prob in enumerate(probabilities):
    print(f"Group {i}: {prob:.4f}")