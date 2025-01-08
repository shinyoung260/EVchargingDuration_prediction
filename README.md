# EV Charging Duration Prediction

This project predicts the duration of electric vehicle (EV) charging sessions using machine learning models. It aims to optimize charging station management by reducing wait times and improving resource utilization.

## Dataset

The dataset used in this project is available on Kaggle: [EV Charging Duration](https://www.kaggle.com/code/anshtanwar/ev-charging-duration).  
Download the dataset and place it in the root directory of the project as `df1_traffic_weather.csv`.

## Workflow

1. **Data Preprocessing**  
   - Import the dataset using `pd.read_csv('df1_traffic_weather.csv')`.  
   - Drop irrelevant columns (`snow_depth`, `revision_status`, `End_plugout`, `Start_plugin_date`).  
   - Convert `El_kWh` and `Duration_hours` to numeric format.
   - One-hot encode categorical columns.
   - Remove outliers using the `IsolationForest` algorithm.
   - Standardize numerical features (`El_kWh` and `Duration_hours`) with `StandardScaler`.

2. **Model Training**  
   - Split the data into training and testing sets (`train_test_split`).
   - Train regression models (`RandomForestRegressor`, `DecisionTreeRegressor`, `Ridge`) to predict `Duration_hours`.
   - Evaluate models using metrics like RMSE.

## Getting Started

1. Install the required libraries:
   ```bash
   pip install pandas numpy matplotlib scikit-learn ydata_profiling
