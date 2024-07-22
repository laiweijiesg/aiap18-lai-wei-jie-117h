# Machine Learning Pipeline for Solar Panel Efficiency Prediction

**Full Name:** Lai Wei Jie  
**Email Address:** wjlai003@mymail.sim.edu.sg

## Overview
This project aims to classify solar panel efficiency as 'Low', 'Medium', or 'High' using historical same-day forecasted weather data and air quality data. The project involves two main tasks:
1. Exploratory Data Analysis (EDA)
2. Development of an end-to-end machine learning pipeline with model selection and fine-tuning.




**Install Dependencies**:
    pip install --upgrade pip
    pip install -r requirements.txt


**Run the Machine Learning Pipeline**:
    ./run.sh

## Folder Structure

│
├── .github
├── src
│   ├── ml.py
│
├── README.MD
├── eda.ipynb
├── requirements.txt
└── run.sh



## Pipeline

1. **Download the Data**:
    - Download the weather and air quality data from the provided URLs.

2. **Load and Merge Data**:
    - Load the data from the SQLite databases and merge them on the `data_ref` key.

3. **Data Preprocessing and Feature Engineering**:
    - Dropped null values.
    - Replaced place holder values "-", "--" with empty strings, and replaced them with median values for each column.
    - Encoded categorical features like 'Dew Point Category', "Daily Solar Panel Efficiency" with LabelEncoder for ordinal values
     and 'Wind Direction with One Hot Encoding for nominal values.
    - Combine PM2.5 and PSI columns into single columns for easier analysis and modeling.
    - Used RobustScaler to scale quantitative data, as data is not normally distributed and might have noise or outliers.

4. **Model Training and Evaluation**:
    - Split the data into training and test sets.
    - Define a pipeline that includes our feature engineering, preprocessing steps, and model training.
    - Train three different models: Random Forest, Gradient Boosting, and Logistic Regression.
    - Perform hyperparameter tuning using GridSearchCV for each model.
    - Evaluate each model using appropriate metrics and save the best models.

## Key Findings from EDA

1. **Missing Values**:
    - Identified and handled missing values by replacing them with the median values of the respective columns.

2. **Categorical Features**:
    - Standardized categorical features such as 'Dew Point Category' and 'Wind Direction' to ensure consistency in the data.

3. **Combined Features**:
    - Combined PM2.5 and PSI values from different directions into single columns (PM25 Combined and PSI Combined) to simplify the data and improve model performance.

4. **Distribution of Target Feature**:
    - Analyzed the distribution of the target feature 'Daily Solar Panel Efficiency', 


## Feature Processing

| Feature                        | Processing Method                                     |
|-------------------------------|------------------------------------------------------|
| Daily Rainfall Total (mm)     | Replace placeholders, Scale using `RobustScaler`     |
| Highest 30 Min Rainfall (mm)  | Replace placeholders, Scale using `RobustScaler`     |
| Highest 60 Min Rainfall (mm)  | Replace placeholders, Scale using `RobustScaler`     |
| Highest 120 Min Rainfall (mm) | Replace placeholders, Scale using `RobustScaler`     |
| Min Temperature (deg C)       | Replace placeholders, Scale using `RobustScaler`     |
| Maximum Temperature (deg C)   | Replace placeholders, Scale using `RobustScaler`     |
| Min Wind Speed (km/h)         | Replace placeholders, Scale using `RobustScaler`     |
| Max Wind Speed (km/h)         | Replace placeholders, Scale using `RobustScaler`     |
| Sunshine Duration (hrs)       | Replace placeholders, Scale using `RobustScaler`     |
| Cloud Cover (%)               | Replace placeholders, Scale using `RobustScaler`     |
| Wet Bulb Temperature (deg F)  | Replace placeholders, Scale using `RobustScaler`     |
| Relative Humidity (%)         | Replace placeholders, Scale using `RobustScaler`     |
| Air Pressure (hPa)            | Replace placeholders, Scale using `RobustScaler`     |
| Dew Point Category            | Encode using `LabelEncoder`                          |
| Wind Direction                | Encode using `OneHotEncoder`                         |
| PM25 Combined                 | Replace placeholders, Scale using `RobustScaler`     |
| PSI Combined                  | Replace placeholders, Scale using `RobustScaler`     |
| Daily Solar Panel Efficiency  | Encode using `LabelEncoder`                          |




## Explanation of Model Choices

1. **Random Forest**:
    - Chosen for its ability to handle large datasets and its robustness to overfitting. It also provides feature importance, which is useful for understanding the impact of different features.

2. **Gradient Boosting**:
    - Selected for its strong predictive performance and ability to handle both numerical and categorical features. It is particularly effective for imbalanced datasets.

3. **Logistic Regression**:
    - Used as a baseline model for its simplicity and interpretability. It is useful for understanding the relationships between features and the target variable.

## Explanation of Evaluation Metrics

1. **Accuracy**:
    - Measures the proportion of correctly classified instances. It is a straightforward metric that provides an overall performance measure.

2. **Classification Report**:
    - Includes precision, recall, and F1-score, which provide a more detailed evaluation of the model's performance, especially for imbalanced classes.

    - Precision:
        Precision is the ratio of correctly predicted positive observations to the total predicted positives. It is also known as the positive predictive value.
        Formula: Precision = TP / (TP + FP)
        Interpretation: High precision indicates a low false positive rate.

    - Recall (Sensitivity or True Positive Rate):
        Recall is the ratio of correctly predicted positive observations to the all observations in the actual class.
        Formula: Recall = TP / (TP + FN)
        Interpretation: High recall indicates a low false negative rate.

    - F1-Score:
        The F1-Score is the harmonic mean of precision and recall. It provides a single metric that balances both the concerns of precision and recall.
        Formula: F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
        Interpretation: The F1-Score is particularly useful when you need to balance precision and recall.

    - Support:
        Support is the number of actual occurrences of the class in the dataset.
        Interpretation: It provides context to the other metrics by showing how many samples of each class are present.



 ### Model Performance

 The best performing model is the Random Forest which works well with both quantitative and qualitative data, and takes into accounts the weight of features. 



Best parameters for Random Forest: {'classifier__max_depth': 30, 'classifier__min_samples_leaf': 1, 'classifier__min_samples_split': 2, 'classifier__n_estimators': 100}
Random Forest Model After Fine-Tuning
              precision    recall  f1-score   support

           0       0.90      0.78      0.84       166
           1       0.91      0.86      0.88       223
           2       0.90      0.96      0.93       463

    accuracy                           0.90       852
   macro avg       0.90      0.87      0.88       852
weighted avg       0.90      0.90      0.90       852

Accuracy: 0.9014084507042254

Best parameters for Gradient Boosting: {'classifier__learning_rate': 0.1, 'classifier__max_depth': 7, 'classifier__n_estimators': 200}
Gradient Boosting Model After Fine-Tuning
              precision    recall  f1-score   support

           0       0.91      0.78      0.84       166
           1       0.91      0.85      0.88       223
           2       0.89      0.97      0.93       463

    accuracy                           0.90       852
   macro avg       0.91      0.87      0.88       852
weighted avg       0.90      0.90      0.90       852

Accuracy: 0.9014084507042254

Best parameters for Logistic Regression: {'classifier__C': 10, 'classifier__penalty': 'l2', 'classifier__solver': 'liblinear'}
Logistic Regression Model After Fine-Tuning
              precision    recall  f1-score   support

           0       0.72      0.34      0.47       166
           1       0.82      0.59      0.69       223
           2       0.71      0.94      0.81       463

    accuracy                           0.73       852
   macro avg       0.75      0.62      0.65       852
weighted avg       0.74      0.73      0.71       852

Accuracy: 0.7312206572769953