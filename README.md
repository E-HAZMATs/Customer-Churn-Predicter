# Customer Churn Predictor — IBM Telco Dataset

Built customer churn predictor models using the [IBM Telco dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) across four algorithms: Decision Tree, Random Forest, XGBoost, and a Neural Network.

## Content

- `utils/helpers.py` — data encoding pipeline and F1 score, both written from scratch
- `sklearn_decision_tree.ipynb` — decision tree with hyperparameter search
- `sklearn_random_forest.ipynb` — random forest with GridSearchCV
- `customer_churn_xgboost.ipynb` — XGBoost
- `customer_churn_classifier_TF_NN.ipynb` — TF neural network with batch norm, dropout, regularization, and keras-tuner for hyperparameter search