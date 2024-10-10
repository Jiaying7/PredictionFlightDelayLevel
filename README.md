# U.S. Flight Delay Prediction

This project predicts flight delay levels in the U.S. using machine learning models based on easily accessible flight data. The dataset includes scheduled flight dates, times, airlines, and departure locations. Three classifiers – Decision Trees, Random Forest, and K-Nearest Neighbour (KNN) – are compared to evaluate which model best predicts flight delay levels.

## Project Overview

Flight delays are a significant issue in the U.S. aviation industry, affecting millions of passengers and airlines annually. This project aims to predict delay levels by leveraging machine learning techniques. By analyzing past flight data, including flight dates, scheduled departure times, departure locations, and operating airlines, we can offer valuable predictions that benefit both passengers and aviation officials.

### Objectives
- Predict flight delays and classify them into delay levels (0-5) based on delay duration.
- Compare the performance of three classifiers (Decision Trees, Random Forest, and K-Nearest Neighbour) to identify the most effective model for this task.
- Provide actionable insights for better decision-making in aviation management.

## Dataset

The dataset used for this project comes from the **Bureau of Transportation Statistics (BTS)**, covering flight data from 2018 to 2023 for various domestic and international flights across multiple airlines in the U.S.

- **Key Features**:
  - Flight Date
  - Operating Airline
  - Origin City
  - Scheduled Departure Time
  - Actual Departure Time
  - Departure Delay (calculated as the difference between scheduled and actual times)

## Machine Learning Models

The project implements and compares the following machine learning models:

### 1. K-Nearest Neighbour (KNN)
- Instance-based learning model that classifies new data points by the majority class of its nearest neighbours.
- Tuned `k` parameter to 5 for optimal performance.

### 2. Decision Tree
- A tree-like structure where each node represents a decision rule, and each leaf node represents a class.
- Tuned parameters include `max_depth` and `max_leaf_nodes`.

### 3. Random Forest
- An ensemble learning method that builds multiple decision trees to improve prediction accuracy.
- Tuned parameters include `max_depth`, `min_samples_leaf`, `max_leaf_nodes`, and `max_features`.

## Methodology

The project follows the **CRISP-DM** (Cross Industry Standard Process for Data Mining) methodology:
1. **Data Collection**: Flight data is collected from BTS, including over 4.5 million records.
2. **Data Pre-processing**: Features such as date, departure time, and airlines are processed. Delay times are categorized into levels from 0 (no delay) to 5 (severe delay).
3. **Model Training**: Models are trained on pre-processed data, with hyperparameter tuning using cross-validation.
4. **Evaluation**: Models are evaluated based on accuracy, precision, recall, and F1-score.

### Flight Delay Levels
The `DEP_DELAY` column is used to classify flight delays into six levels:
- **0**: No delay (<15 minutes)
- **1**: Moderate delay (15 to <45 minutes)
- **2**: Delays between 45 to <75 minutes
- **3**: Delays between 75 to <105 minutes
- **4**: Delays between 105 to <135 minutes
- **5**: Severe delay (135+ minutes)

## Results

After hyperparameter tuning, the **Decision Tree** algorithm demonstrated the highest accuracy at **78.7%**, followed closely by Random Forest and KNN.

- **Decision Tree**: 78.7% accuracy, 69.89% F1-score
- **Random Forest**: 78.66% accuracy, 69.42% F1-score
- **KNN**: 77.14% accuracy, 71% F1-score

### Feature Importance
The most important features in predicting flight delays were:
1. **Flight Date**
2. **Scheduled Departure Time**
3. **Operating Carrier**
4. **Origin City**

## How to Run

1. **Clone the repository**:
    ```bash
    git clone https://github.com/YOUR_USERNAME/flight-delay-prediction.git
    ```
2. **Install dependencies**:
    - Install Python and the required libraries:
      ```bash
      pip install -r requirements.txt
      ```
3. **Run the model**:
    ```bash
    python flight_delay_prediction.py
    ```

4. **Evaluate the model**:
    - Check the model's performance and predictions using the included evaluation scripts.

## Technologies Used

- **Python**: Main programming language.
- **scikit-learn**: Machine learning models and evaluation tools.
- **pandas**: Data manipulation and preprocessing.
- **matplotlib**: Visualization of results.
- **SMOTE/ADASYN**: Oversampling techniques for handling imbalanced data.

## Future Work

- **Improving Imbalance Handling**: Implement better data augmentation techniques to handle the imbalance in flight delay levels.
- **Integration of Weather Data**: Including weather information for more accurate predictions of flight delays.
- **Extending the Dataset**: Expand the dataset to include international flights for a more global application.

## License

This project is licensed under the MIT License.
