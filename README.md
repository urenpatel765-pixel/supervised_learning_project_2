# 🏠 House Price Prediction Using Supervised Learning

## 📌 Project Overview

This project focuses on predicting house prices using supervised machine learning regression techniques.

The objective is to build, evaluate, compare, and select the most suitable regression model for predicting `house_price_inr` from different property-related features.

The project follows a complete machine learning workflow, starting from dataset understanding and preprocessing, followed by regularized linear regression, cross-validation, tree-based regression, Support Vector Regression, model comparison, residual analysis, and final model selection.

The final selected model is **Random Forest**, which achieved the strongest performance on the testing dataset.

---

# 🎯 Project Objectives

The main objectives of this project are:

- 📊 Understand and inspect the house-price dataset.
- 🔍 Perform dataset exploration and quality checks.
- 🧹 Identify missing values and duplicate records.
- 🎯 Define the target variable and predictor features.
- ✂️ Split the dataset into training and testing datasets.
- ⚙️ Apply appropriate preprocessing and feature transformation.
- 📈 Build regularized regression models using Ridge and Lasso.
- 🔄 Compare different cross-validation strategies.
- 🌳 Build Decision Tree and Random Forest regression models.
- ⚙️ Tune model hyperparameters.
- 🤖 Apply Support Vector Regression with different kernels.
- 📊 Compare all developed models using MSE, MAE, RMSE, and R2.
- 📉 Perform residual analysis.
- 🏆 Select the best-performing final model.
- 📝 Interpret the results from a real-world business perspective.

---

# 🗂️ Dataset Information

The dataset contains **3,800 property records** and **12 columns**.

### Dataset Shape

```text
Rows    : 3800
Columns : 12
Dataset Features
Feature	Description
property_id	Unique property identifier
sale_date	Date of property sale
area_sqft	Property area in square feet
bedrooms	Number of bedrooms
bathrooms	Number of bathrooms
location_score	Location quality score
property_age	Age of the property
distance_city_km	Distance from the city
near_school	Indicates proximity to a school
near_metro	Indicates proximity to metro
crime_rate_index	Crime-rate index of the area
house_price_inr	Target house price in INR
🧠 Machine Learning Problem
Problem Type

Supervised Learning → Regression

Target Variable
house_price_inr
Number of Features
9
Features Used
area_sqft
bedrooms
bathrooms
location_score
property_age
distance_city_km
near_school
near_metro
crime_rate_index

📚 PART A — DATASET UNDERSTANDING & EXPLORATION

The dataset was first inspected to understand its structure, dimensions, column names, data types, statistical properties, missing values, and duplicate records.

Dataset Validation Results
Dataset Shape: (3800, 12)
Duplicate Records: 0
Missing Values: 0
Target Variable: house_price_inr
Number of Features: 9
Number of Records: 3800

All 3,800 records were checked for missing values and duplicate records.

The analysis showed that there were no missing values and no duplicate records in the dataset.

The target variable was identified as house_price_inr, while nine property-related variables were used as predictive features.

✂️ PART B — DATA PREPARATION & SPLITTING

The dataset was divided into training and testing subsets.

Dataset Split
Training Data : 80%
Testing Data  : 20%
Result
Training Features : (3040, 9)
Training Target   : (3040,)

Testing Features  : (760, 9)
Testing Target    : (760,)

The 80:20 split ensures that the models are trained using the majority of the available observations while keeping an unseen testing dataset for final evaluation.

⚙️ PART C — REGULARIZED LINEAR REGRESSION

Two regularized linear regression techniques were implemented:

1. Ridge Regression

Ridge Regression applies L2 regularization to control the size of model coefficients and reduce model complexity.

2. Lasso Regression

Lasso Regression applies L1 regularization. It can shrink coefficients toward zero and can also perform feature selection when coefficients become exactly zero.

🔧 Hyperparameter Selection

Different alpha values were evaluated using cross-validation.

Best Parameters
Model	Best Alpha
Ridge Regression	1
Lasso Regression	100

Lasso produced a slightly lower validation RMSE than Ridge.

📊 Regularized Regression Results

Model	Test RMSE	Test MAE	Test R2
Ridge Regression	2,558,402	1,959,199	0.918726
Lasso Regression	2,558,124	1,959,378	0.918744
Observation

Lasso achieved a slightly lower testing RMSE and slightly higher testing R2 than Ridge.

The Lasso model did not reduce any coefficient completely to zero:

Lasso coefficients reduced to zero: 0

Therefore, Lasso performed coefficient shrinkage without completely removing any of the nine features.

🔄 PART D — CROSS-VALIDATION & MODEL STABILITY

Cross-validation was used to evaluate model performance across different data splits.

The following strategies were compared:

K-Fold
Stratified K-Fold
Leave-One-Out
Time Series Split

Cross-validation provides additional information about model stability instead of relying only on one train-test split.

📊 Cross-Validation Results

CV Strategy	RMSE	R2
K-Fold	2,511,687	0.915225
Stratified K-Fold	2,508,705	0.915547
Leave-One-Out	2,509,656	0.915578
Time Series Split	2,511,495	0.915071
Result
Lowest Overall RMSE:
Stratified K-Fold

Most Stable CV Strategy:
Stratified K-Fold

The reported results show that Stratified K-Fold produced the lowest overall RMSE among the compared strategies.

📷 Graph 1 — Comparison of Cross-Validation Strategies

Insert the actual graph screenshot generated by the project code below.

![Figure 1 - Cross-Validation Strategy Comparison](https://github.com/user-attachments/assets/d48d72a4-10fa-4bdd-a882-e28ff707e88d)



Figure 1: Comparison of Cross-Validation Strategies

📈 REGULARIZATION ANALYSIS

The effect of different regularization strengths was also analyzed by comparing Ridge and Lasso validation RMSE across different alpha values.

The analysis showed that the models remained relatively stable across smaller alpha values, while stronger regularization affected the validation error differently for Ridge and Lasso.

📷 Graph 2 — Regularization Strength vs Validation Error

Insert the actual graph screenshot generated by the project code below.

![Figure 2 - Regularization Strength vs Validation Error](https://github.com/user-attachments/assets/7353b40f-2a12-448e-8ec9-047f17c00308)


Figure 2: Regularization Strength vs Validation Error

🌳 PART E — TREE-BASED REGRESSION

Tree-based regression models were implemented to capture non-linear relationships between property characteristics and house prices.

The following models were evaluated:

Decision Tree
Tuned Decision Tree
Random Forest
🌲 Decision Tree

The initial Decision Tree achieved:

Training R2 : 1.000000
Testing R2  : 0.858216

The training score of 1.0 compared with the lower testing score indicates clear overfitting.

⚙️ Tuned Decision Tree

Hyperparameter tuning was performed to improve the generalization performance of the Decision Tree.

Best Parameters
max_depth = 7
min_samples_leaf = 5
min_samples_split = 2
Tuned Decision Tree Performance
Training RMSE : 2,203,180
Testing RMSE  : 2,769,337

Training R2   : 0.934938
Testing R2    : 0.904772

The tuned model reduced the generalization problem compared with the original Decision Tree.

🌲 Random Forest

Random Forest combines multiple decision trees and averages their predictions to improve predictive performance and reduce the limitations of an individual tree.

Random Forest Performance
Training RMSE : 887,509
Testing RMSE  : 2,408,752

Training R2   : 0.989442
Testing R2    : 0.927956

Random Forest achieved the best testing performance among the tree-based models.

📊 Tree-Based Model Comparison
Model	Test RMSE	Test MAE	Test R2
Decision Tree	3,379,145	2,480,762	0.858216
Tuned Decision Tree	2,769,337	2,013,210	0.904772
Random Forest	2,408,752	1,772,057	0.927956
🏆 Best Tree-Based Model
Random Forest
Test RMSE : INR 2,408,751.74
Test R2   : 0.9280
📷 Graph 3 — Decision Tree vs Random Forest Performance

Insert the actual graph screenshot generated by the project code below.

![Figure 3 - Decision Tree vs Random Forest Performance](https://github.com/user-attachments/assets/cfd3d51d-f71f-4b08-8e53-d4b56313ed16)


Figure 3: Decision Tree vs Random Forest Performance

🤖 PART F — SUPPORT VECTOR REGRESSION

Support Vector Regression was implemented using different kernels to investigate whether kernel-based regression could provide better predictions for the dataset.

The following kernels were evaluated:

Linear
Polynomial
RBF

A tuned SVR model was also evaluated.

📊 SVR Kernel Results
Model	Test RMSE	Test MAE	Test R2
Linear SVR	8,982,722	6,990,272	-0.001914
Polynomial SVR	8,987,085	6,993,875	-0.002887
RBF SVR	8,987,327	6,994,024	-0.002941
Tuned SVR	8,521,614	6,620,813	0.098308
Best SVR Result
Best Overall SVR Model: Tuned SVR
Test RMSE : INR 8,521,614.19
Test R2   : 0.0983

Tuning improved the SVR performance compared with the untuned kernels. However, the SVR models performed substantially worse than the linear and tree-based models on this dataset.

📷 Graph 4 — SVR Kernel Performance Comparison

Insert the actual graph screenshot generated by the project code below.

![Figure 4 - SVR Kernel Performance Comparison](https://github.com/user-attachments/assets/c758ea76-212a-4bf0-993a-cf95a7662fc3)

Figure 4: SVR Kernel Performance Comparison

🏆 PART G — FINAL MODEL COMPARISON

All major regression models were compared using their testing performance.

The evaluation metrics used were:

MSE — Mean Squared Error
MAE — Mean Absolute Error
RMSE — Root Mean Squared Error
R2 — Coefficient of Determination

For model selection, lower RMSE and MAE indicate lower prediction error, while a higher R2 indicates stronger explanatory performance.

📊 Final Model Comparison
Rank	Model	Test RMSE	Test MAE	Test R2
🥇 1	Random Forest	2,408,752	1,772,057	0.927956
🥈 2	Lasso Regression	2,558,124	1,959,378	0.918744
🥉 3	Ridge Regression	2,558,402	1,959,199	0.918726
4	Tuned Decision Tree	2,769,337	2,013,210	0.904772
5	Decision Tree	3,379,145	2,480,762	0.858216
6	Tuned SVR	8,521,614	6,620,813	0.098308
🥇 FINAL MODEL: RANDOM FOREST

Based on the testing results, Random Forest was selected as the final model.

Final Performance
Test RMSE : INR 2,408,751.74
Test MAE  : INR 1,772,057.16
Test R2   : 0.9280

The Random Forest model achieved the lowest testing RMSE among the evaluated models and the highest testing R2.

Therefore, it provided the strongest overall predictive performance on the unseen testing data.

📉 GENERALIZATION ANALYSIS

Training and testing R2 values were compared to identify possible overfitting or underfitting.

Model	Training R2	Testing R2	R2 Gap
Ridge Regression	0.916185	0.918726	-0.002541
Lasso Regression	0.916186	0.918744	-0.002558
Decision Tree	1.000000	0.858216	0.141784
Tuned Decision Tree	0.934938	0.904772	0.030166
Random Forest	0.989442	0.927956	0.061486
Tuned SVR	0.097414	0.098308	-0.000894
Observations
🌳 The original Decision Tree showed the largest generalization gap and clear overfitting.
⚙️ Hyperparameter tuning improved the Decision Tree's generalization.
🌲 Random Forest achieved the strongest testing performance.
📈 Ridge and Lasso showed stable training and testing performance.
🤖 SVR showed comparatively weak predictive performance for this dataset.
📷 Graph 5 — Actual vs Predicted House Prices

The Actual vs Predicted graph compares the real house prices with the prices predicted by the final Random Forest model.

Points closer to the reference diagonal indicate predictions closer to the actual values.

The graph provides a visual confirmation of the strong relationship between actual and predicted house prices.

Insert the actual graph screenshot generated by the project code below.

![Figure 5 - Actual vs Predicted House Prices](https://github.com/user-attachments/assets/334cf5b2-21e4-4294-9541-8ac0f274c540)


Figure 5: Actual vs Predicted House Prices

📉 RESIDUAL ANALYSIS

Residual analysis was performed to examine the prediction errors of the final model.

A residual represents the difference between an actual value and its predicted value.

The final model produced the following residual statistics:

Mean Residual:
INR 67,913.39

Mean Absolute Error:
INR 1,772,057.16

Residual Standard Deviation:
INR 2,407,794.16

The residual plot helps identify the distribution of prediction errors and possible patterns in the model's predictions.

📷 Graph 6 — Residuals vs Predicted House Prices

Insert the actual graph screenshot generated by the project code below.

 ![Figure 6 - Residuals vs Predicted House Prices](https://github.com/user-attachments/assets/a1d69257-2141-4446-938c-cb55fc4a35ca)

Figure 6: Residuals vs Predicted House Prices

💼 BUSINESS INTERPRETATION

The developed model can be used as a decision-support system for estimating property prices based on available property characteristics.

Important input factors include:

🏠 Property area
🛏️ Number of bedrooms
🛁 Number of bathrooms
📍 Location score
🏚️ Property age
🚗 Distance from city
🏫 Proximity to school
🚇 Proximity to metro
🛡️ Crime-rate index

The final Random Forest model achieved an R2 of approximately 0.9280 on the testing dataset.

This indicates a strong predictive relationship between the available property features and the observed house prices in this dataset.

The model can potentially assist real-estate businesses, property analysts, buyers, and sellers in obtaining an estimated property value.

However, actual market prices may also depend on additional factors that are not included in the dataset, such as market demand, exact neighborhood conditions, property condition, amenities, and economic conditions.

Therefore, the model should be considered a decision-support tool rather than a replacement for professional property valuation.

🔬 KEY FINDINGS
📌 Finding 1 — Dataset Quality

The dataset contained:

3,800 records
12 columns
0 missing values
0 duplicate records
📌 Finding 2 — Regularization

Lasso achieved slightly better testing performance than Ridge.

Lasso Test R2 : 0.918744
Ridge Test R2 : 0.918726
📌 Finding 3 — Cross-Validation

Among the compared strategies, Stratified K-Fold produced the lowest reported overall RMSE.

Stratified K-Fold RMSE : approximately 2.51 million
📌 Finding 4 — Decision Tree

The original Decision Tree showed clear overfitting.

Training R2 : 1.000000
Testing R2  : 0.858216
📌 Finding 5 — Random Forest

Random Forest provided the best overall testing performance.

Test RMSE : INR 2,408,751.74
Test MAE  : INR 1,772,057.16
Test R2   : 0.9280
📌 Finding 6 — SVR

SVR performed considerably worse than the other evaluated approaches.

Tuned SVR Test R2 : 0.0983
🏆 FINAL CONCLUSION

This project successfully implemented a complete supervised learning workflow for house-price prediction.

The analysis began with dataset exploration and validation, followed by feature selection, train-test splitting, preprocessing, regularized regression, cross-validation, tree-based regression, Support Vector Regression, hyperparameter tuning, model comparison, and residual analysis.

Ridge and Lasso demonstrated strong and stable linear-model performance through regularization.

The original Decision Tree showed overfitting, while hyperparameter tuning improved its generalization.

Random Forest provided the strongest overall predictive performance among the evaluated models.

🥇 Final Selected Model

Random Forest

📊 Final Performance
RMSE : INR 2,408,751.74
MAE  : INR 1,772,057.16
R2   : 0.9280

Therefore, Random Forest was selected as the final model because it achieved the lowest testing RMSE and highest testing R2 among the evaluated models.

📁 PROJECT DELIVERABLES

The project submission contains:

📓 Complete Jupyter Notebook / Source Code
📊 Dataset
📈 Model evaluation tables
📉 Residual analysis
📊 Six generated visualizations
🔄 Cross-validation analysis
⚙️ Hyperparameter tuning results
🏆 Final model comparison
📝 Final analysis and conclusion
📊 VISUALIZATION INDEX
Figure	Graph	Purpose
1	Cross-Validation Strategy Comparison	Compare validation strategies
2	Regularization Strength vs Validation Error	Analyze Ridge and Lasso
3	Decision Tree vs Random Forest Performance	Compare tree-based models
4	SVR Kernel Performance Comparison	Compare SVR kernels
5	Actual vs Predicted House Prices	Visualize prediction performance
6	Residuals vs Predicted House Prices	Analyze prediction errors
🛠️ TECHNOLOGIES & LIBRARIES
Programming Language

🐍 Python

Development Environment

📓 Jupyter Notebook / VS Code

Main Libraries
pandas — Data manipulation and analysis
numpy — Numerical computation
matplotlib — Data visualization
scikit-learn — Machine learning models, preprocessing, validation, and evaluation
Machine Learning Techniques
Linear Regression
Ridge Regression
Lasso Regression
K-Fold Cross-Validation
Stratified K-Fold
Leave-One-Out Cross-Validation
Time Series Split
Decision Tree Regression
Random Forest Regression
Support Vector Regression
Hyperparameter Tuning
Residual Analysis
🎓 PROJECT LEARNING OUTCOMES

Through this project, the following concepts were practically implemented:

Understanding supervised learning regression
Dataset exploration and validation
Feature and target identification
Train-test splitting
Data preprocessing
Regularization
Hyperparameter tuning
Cross-validation
Linear regression
Non-linear regression
Tree-based ensemble learning
Support Vector Regression
Model evaluation
Overfitting detection
Generalization analysis
Residual analysis
Model comparison
Business interpretation of machine learning results
✅ PROJECT STATUS
🎉 COMPLETED

The supervised learning house-price prediction project has completed the required stages from dataset analysis through final model selection and reporting.

Final Model

🏆 Random Forest Regression

Final Test Performance

RMSE: INR 2,408,751.74
MAE: INR 1,772,057.16
R2: 0.9280

📌 FINAL SUBMISSION NOTE

This README provides the documented analysis and interpretation of the implemented supervised learning project.

The six graphs included in the visualization section were generated directly during execution of the project code and should be inserted at their corresponding positions above.

Each graph should be included only once to maintain a clear and professional project presentation.

👨‍💻 PROJECT SUBMISSION

Project: House Price Prediction Using Supervised Learning
Task: Regression Model Development and Evaluation
Final Model: Random Forest Regression
Dataset Size: 3,800 Records
Final Test R2: 0.9280
Final Test RMSE: INR 2,408,751.74

⭐ END OF PROJECT REPORT

Thank You! 🙏


### 📌 One important thing before submission

For the **six placeholders**, put your screenshots in this exact order:

1. **Comparison of Cross-Validation Strategies**
2. **Regularization Strength vs Validation Error**
3. **Decision Tree vs Random Forest Performance**
4. **SVR Kernel Performance Comparison**
5. **Actual vs Predicted House Prices**
6. **Residuals vs Predicted House Prices**

That gives you a clean **one-screenshot-per-result** structure and makes it much easier for an evaluator to see that every major requirement has visual evidence.



