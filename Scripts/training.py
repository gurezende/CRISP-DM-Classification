# -------- Imports --------
import pandas as pd
import numpy as np
import mlflow

# Transformations and Modeling
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report
from catboost import CatBoostClassifier, Pool
from my_funcs import *

import warnings
warnings.filterwarnings('ignore')


# -------- Load the Data --------
from ucimlrepo import fetch_ucirepo

# fetch dataset
bank_marketing = fetch_ucirepo(id=222)

# data (as pandas dataframes)
df = pd.concat([bank_marketing.data.features, bank_marketing.data.targets], axis=1)
df = df.rename(columns={'day_of_week':'day'})

# Split in train and test sets
X_train, X_test, y_train, y_test = train_test_split(df.drop('y', axis=1),
                                                    df['y'],
                                                    test_size=0.2,
                                                    stratify=df['y'],
                                                    random_state=42)

# train
df_train = pd.concat([X_train, y_train], axis=1)

# test
df_test = pd.concat([X_test, y_test], axis=1)


# -------- Prepare the Data --------

# Selected columns for the model
cols_order = ['age',  'job_admin.', 'job_services', 'job_management', 'job_blue-collar', 'job_unemployed', 'job_student', 'job_technician',
                                    'contact_cellular', 'contact_telephone', 'job_retired', 'poutcome_failure', 'poutcome_other', 'marital_single', 'marital_divorced',
                                    'previous', 'pdays', 'campaign', 'month', 'day', 'loan', 'housing', 'default', 'poutcome_unknown', 'y']

# Preparing data for predictions
# X_train, y_train = prepare_data_select_columns(df_train, cols_order)
# X_test, y_test = prepare_data_select_columns(df_test, cols_order)

# Preparing data for simpler model
X_train, y_train = prepare_data_simpler_model(df_train)
X_test, y_test = prepare_data_simpler_model(df_test)


# -------- Connect to MLFlow --------

# Connecting with MLFlow Server
mlflow.set_tracking_uri("http://127.0.0.1:5000/")

# Set Experiment ID (Get from MLFlow Experiment Tab)
mlflow.set_experiment(experiment_id=901255489418492369) #using separate env for mlflow


# ---------- Training the model ------------

# Set the description as a tag
description = "Simpler Model Classifier"

# Start MLFlow Run:
with mlflow.start_run(description=description):
    
    # Fit
    model = (
        CatBoostClassifier(iterations=300,
                           learning_rate=0.1,
                           depth = 5,
                           loss_function='Logloss',
                           border_count= 64,
                           l2_leaf_reg= 13,
                           class_weights=[1, 3],
                           early_stopping_rounds=50,
                           verbose=1000)
        .fit(X_train, y_train)
    )

    # Ask MLFlow to log the basic information about the model
    mlflow.catboost.log_model(cb_model=model,
                              input_example=X_train,
                              artifact_path="simpler_model")

    # train the model
    model.fit(X_train, y_train)

    # Predict
    prediction = model.predict(X_test)

    # Evaluate the model
    f1score = f1_score(y_test, prediction)
    accuracy = accuracy_score(y_test, prediction)
    cm = confusion_matrix(y_test, prediction)
    report = classification_report(y_test, prediction)
    
    # Log the metrics
    mlflow.log_metric("f1_score", f1score)
    mlflow.log_metric("accuracy", accuracy)
    


    # -------- Voting Classifier ---------
    
    # Imports
    # from sklearn.ensemble import VotingClassifier
    # from sklearn.neighbors import KNeighborsClassifier
    # from sklearn.tree import DecisionTreeClassifier
    # from sklearn.ensemble import GradientBoostingClassifier
    # from xgboost import XGBClassifier

    # # Creating instances of the algorithms
    # catboost = CatBoostClassifier(iterations=300, depth=5, learning_rate=0.1, loss_function='Logloss', class_weights=[1,3],
    #                             border_count= 64, l2_leaf_reg= 13, early_stopping_rounds=50, verbose=False)
    
    # # Instance the models
    # knn = KNeighborsClassifier(n_neighbors=20)
    # dt = DecisionTreeClassifier(max_depth=4, class_weight='balanced')
    # gradboost = GradientBoostingClassifier()
    # xgb = XGBClassifier(scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]))

    # # Voting Classifier
    # voting = VotingClassifier(estimators=[
    #         ('catboost', catboost),
    #         ('knn', knn),
    #         ('dt', dt),
    #         ('gradboost', gradboost),
    #         ('xgb', xgb)
    #         ],
    #         voting='soft')

    # # Fit
    # voting.fit(X_train, y_train)

    # # Predict
    # y_pred = voting.predict(X_test)

    # # Evaluate the model
    # f1score = f1_score(y_test, y_pred)
    # accuracy = accuracy_score(y_test, y_pred)
    
    # # Log the metrics
    # mlflow.log_metric("f1_score", f1score)
    # mlflow.log_metric("accuracy", accuracy)