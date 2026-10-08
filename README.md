# 🚗 Car Price Prediction Using Machine Learning

A machine learning regression project that predicts the selling price of used cars based on features such as manufacturing year, present price, kilometers driven, fuel type, seller type, transmission, and number of previous owners.

The project compares multiple regression algorithms and selects the final model using **5-fold cross-validation**. The trained model is also deployed as an interactive **Streamlit web application**.

## 📌 Project Overview

The objective of this project is to develop a machine learning model capable of estimating the selling price of a used car from its available features.

Three regression algorithms were implemented and compared:

* Linear Regression
* Random Forest Regressor
* Support Vector Regression (SVR)

The final model was selected based on the lowest **mean 5-fold cross-validation RMSE**.

## 📊 Dataset

The project uses a used-car dataset containing information about:

* Manufacturing year
* Present market price
* Kilometers driven
* Fuel type
* Seller type
* Transmission type
* Number of previous owners
* Selling price

The `Car_Name` feature was removed during preprocessing because it is a high-cardinality identifier rather than a useful generalized numerical/categorical predictor.

## ⚙️ Data Preprocessing

The following preprocessing steps were performed:

1. Duplicate rows were removed.
2. The `Car_Name` column was removed.
3. Features and target variable were separated.
4. Numerical features were standardized using `StandardScaler`.
5. Categorical features were encoded using `OneHotEncoder`.
6. Preprocessing was implemented inside a Scikit-learn pipeline to prevent data leakage.
7. The dataset was divided into training and testing sets using an 80:20 split.

## 🤖 Machine Learning Models

### 1. Linear Regression

Used as a baseline regression model to establish a simple relationship between the input features and selling price.

### 2. Random Forest Regressor

An ensemble learning algorithm that combines multiple decision trees. It can model nonlinear relationships between car features and selling price.

### 3. Support Vector Regression

SVR with an RBF kernel was used to model potentially nonlinear relationships in the dataset.

## 📈 Model Evaluation

The models were evaluated using:

* **MAE (Mean Absolute Error)** — measures the average absolute prediction error.
* **RMSE (Root Mean Squared Error)** — penalizes larger prediction errors more strongly.
* **R² Score** — measures the proportion of variation in the target explained by the model.

### Test Set Results

| Model             |   MAE |  RMSE | R² Score |
| ----------------- | ----: | ----: | -------: |
| Linear Regression | 1.473 | 2.524 |    0.753 |
| Random Forest     | 1.500 | 3.581 |    0.503 |
| SVR               | 1.190 | 3.076 |    0.633 |

### 5-Fold Cross-Validation

| Model             | Mean CV RMSE | Std CV RMSE |
| ----------------- | -----------: | ----------: |
| Linear Regression |        1.756 |       0.536 |
| Random Forest     |        1.521 |       0.938 |
| SVR               |        2.759 |       1.258 |

Based on the **lowest mean 5-fold cross-validation RMSE**, the **Random Forest Regressor** was selected as the final model.

### Final Model Performance

**Random Forest Regressor**

* MAE: **1.500**
* RMSE: **3.581**
* R² Score: **0.503**

## 📊 Visualizations

The project generates the following visualizations:

### Selling Price Distribution

Shows the distribution of car selling prices in the dataset.

### Correlation Heatmap

Shows correlations between numerical features.

### Actual vs Predicted Prices

Compares the actual selling prices with the prices predicted by the final model.

### Residual Plot

Shows the prediction errors of the final Random Forest model.

### Model Comparison

Compares the models using their mean 5-fold cross-validation RMSE.

## 🌐 Streamlit Application

The trained Random Forest model is deployed through a Streamlit web application.

The application allows users to enter car details and receive an estimated selling price.

### Run the application

Install the required packages:

```bash
pip install -r requirements.txt
```

Then run:

```bash
streamlit run app.py
```

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib
* Streamlit
* Git & GitHub

## 📁 Project Structure

```text
Car_Price_Prediction/
│
├── app.py
├── car data.csv
├── car_price_prediction.py
├── car_price_random_forest.pkl
├── README.md
├── .gitignore
│
└── plots/
    ├── selling_price_distribution.png
    ├── correlation_heatmap.png
    ├── actual_vs_predicted.png
    ├── residual_plot.png
    └── model_comparison_cv_rmse.png
```

## ⚠️ Limitations

* The dataset is relatively small.
* The model's performance depends on the quality and range of the available training data.
* Used-car prices can be influenced by factors not included in the dataset, such as vehicle condition, location, maintenance history, and market demand.
* The model should be considered an estimation tool rather than a replacement for professional vehicle valuation.

## 🎯 Learning Outcomes

This project provided practical experience in:

* Data preprocessing
* Regression modelling
* Feature encoding and scaling
* Machine learning pipelines
* Model evaluation
* Cross-validation
* Data visualization
* Model deployment using Streamlit
* Git and GitHub version control

## 👩‍💻 Author

**Ashitha**

B.Tech Computer Science and Engineering

---

⭐ If you find this project useful, feel free to explore the code and visualizations.
