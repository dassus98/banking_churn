# banking_churn

<br>

# **Banking Churn Analysis and Prediction**

This project explores customer churn behavior in a banking dataset and predicts churn using advanced machine learning models. 10,000 customer records were analyzed for identifiable patterns and predictive models were built to provide actionable insights to minimize churn rates. The models include **Decision Trees**, **Random Forests** and **Feedforward Neural Networks** which were all optimized for high performance.

---

## **Table of Contents**
1. [Project Overview](#project-overview)
2. [Dataset Description](#dataset-description)
3. [Technologies Used](#technologies-used)
4. [EDA Insights](#eda-insights)
5. [Modeling Approach](#modeling-approach)
6. [Results](#results)
7. [How to Run](#how-to-run)
8. [Future Work](#future-work)
9. [Contact](#contact)

---

## **Project Overview**
Customer churn is a critical challenge in the banking industry where retaining customers is more cost-effective than acquiring new ones. This project provides:
- In-depth **Exploratory Data Analysis (EDA)** to identify churn patterns.
- Predictive models to classify customers likely to churn.
- Insights into key drivers of churn for targeted intervention.

---

## **Dataset Description**
The dataset contains 10,000 rows of customer data including:
- **Demographics**: Gender, Geography, Age.
- **Account Information**: Balance, Number of Products, Credit Score.
- **Churn Labels**: Whether the customer left the bank (1 = Churn, 0 = Not Churn).

Source: [Bank Customer Churn Prediction Dataset](https://www.kaggle.com/datasets/saurabhbadole/bank-customer-churn-prediction-dataset)

---

## **Libraries Used**
  - Data Analysis: `pandas`, `numpy`
  - Visualization: `matplotlib`, `seaborn`
  - Machine Learning: `scikit-learn`, `keras`, `TensorFlow`
- **Tools Used**: Jupyter Notebook

---

## **EDA Insights**
Key insights from the Exploratory Data Analysis include:
- **High Risk Demographics**:
  - Customers with low tenure are more likely to churn.
  - Geography influences churn rates significantly.
- **Account Features**:
  - High account balance correlates with lower churn rates.
  - Customers with fewer products are at higher risk of leaving.

---

## **Modeling Approach**
1. **Data Preprocessing**:
   - Handled missing values.
   - Encoded categorical variables using OneHotEncoding.
   - Scaled numeric features using `StandardScaler`.

2. **Models Used**:
   - Each modelling technique was explored as both a raw Base model and a tuned Final model. 
   - **Decision Tree**:
     - Baseline model with a validation accuracy of 79.1%.
   - **Random Forest**:
     - Baseline model with a validation accuracy of 85.4%.
   - **Neural Network**:
     - Baseline model with a validation accuracy of 86.5%.

3. **Evaluation Metrics**:
   - Accuracy
   - Precision
   - Recall
   - F1-Score

---

## **Results**
| Model             | Accuracy | Precision | Recall | F1-Score |
|--------------------|----------|-----------|--------|----------|
| Decision Tree      | 86.5%      | 78.5%       | 46.6%    | 58.5%      |
| Random Forest      | 87.0%      | 83.6%       | 45.1%    | 58.6%      |
| Neural Network 1   | 86.0%      | 77.2%       | 38.2%    | 51.2%      |
| Neural Network 2   | 85.5%      | 83.3%       | 31.9%    | 46.1%      |

Key drivers of churn identified:
- **Tenure**: Short tenure increases churn likelihood.
- **Number of Products**: Customers with only one product are more likely to churn.
- **Geography**: Customers from certain regions show higher churn rates.

---

## **How to Run**
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/banking-churn-analysis.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook Banking_Churn_Analysis.ipynb
   ```

---

## **Future Work**
- Add **Explainable AI** (XAI) tools like SHAP for Neural Networks.
- Deploy the model using Flask or Streamlit for real-time predictions.
- Test additional models like Gradient Boosting (e.g., XGBoost).

---

## **Contact**
Feel free to reach out for questions or collaboration:
- **Email**: cdas@uwaterloo.ca
- **LinkedIn**: [John Chitrakut Das](https://www.linkedin.com/in/chitrakut-das-4b615724b/)
