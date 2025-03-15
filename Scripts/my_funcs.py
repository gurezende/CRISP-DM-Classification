## Functions for the project CRISP-DM Notebook

import pandas as pd
import numpy as np
from feature_engine.encoding import OneHotEncoder
import category_encoders as ce

# Instance
ohe = OneHotEncoder(variables=['marital', 'poutcome', 'education', 'contact', 'job', 'balance'], drop_last=True)

# Function to replace the variable data with the new categorized bins
def variable_to_category(data, variable, k):
  """
  Converts a continuous variable in a DataFrame into categorical bins.

  Parameters:
  data (pd.DataFrame): The DataFrame containing the data.
  variable (str): The name of the column in the DataFrame to be categorized.
  k (int): The number of bins to categorize the variable into.

  Returns:
  pd.Series: A Series object with the categorized data as strings.
  """

  return pd.cut(data[variable], bins=k).astype(str)


# Function to prepare the data for prediction
def prepare_data_all_columns(dtf):
  '''
  Function to prepare the data for prediction. The steps are the same used to prepare the training data.
  * Input:
  dtf: dataframe

  * Output:
  data clean: dataframe with the cleaned data
  '''

  # Imports
  from feature_engine.encoding import OneHotEncoder
  from feature_engine.imputation import CategoricalImputer
  import category_encoders as ce

  # Drop NAs from variable jobs
  dtf = dtf.dropna(subset='job')

  # Simple Imputer for education
  imputer = CategoricalImputer(
      variables=['education'],
      imputation_method="frequent"  )

  # Fit and Transform
  imputer.fit(dtf)
  dtf = imputer.transform(dtf)

  # Input "unknown" for NAs in poutcome.
  dtf['poutcome'] = dtf['poutcome'].fillna('unknown')

  # Fill NAs with "unknown" in contact
  dtf['contact'] = dtf['contact'].fillna('unknown')

  # Binarizing default, housing, loan, and y
  dtf = dtf.replace({'no': 0, 'yes': 1})

  # Variable balance in 3 categories: <0 = 'negative, 0-median = 'avg', >median = 'over avg'
  dtf = (
      dtf
      .assign(balance = lambda x: np.where(x.balance < 0,
                                            'negative',
                                            np.where(x.balance < x.balance.median(),
                                                    'avg',
                                                    'over avg')
                                            )       )
  )

  # Instance OHR
  ohe = OneHotEncoder(variables=['marital', 'poutcome', 'education', 'contact', 'job', 'balance'], drop_last=True)

  # Fit
  ohe.fit(dtf)

  # Transform
  dtf = ohe.transform(dtf)

  # Move y to the first column
  dtf.insert(0, 'y', dtf.pop('y'))

  # Month to numbers
  dtf['month'] = dtf['month'].map({ 'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6, 'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12})

  # drop variable duration
  dtf = dtf.drop('duration', axis=1)

  # Transforming variable Age into bins
  # Using Sturges rule, where number of bins k = 1 + 3.3*log10(n)
  k = int( 1 + 3.3*np.log10(len(dtf)) )

  # Categorize age, balance, duration, previous, pdays
  for var in str.split('age,pdays,previous', sep=','):
    dtf[var] = variable_to_category(dtf, var, k=k)

  # CatBoost Encoding the dataset
  dtf = ce.CatBoostEncoder().fit_transform(dtf, dtf['y'])

  # Reindex to match columns from the fitted model
  # cols_order = ['y','default', 'housing', 'loan', 'day', 'contact_cellular', 'contact_telephone', 'month', 'campaign', 'pdays']

  cols_order = ['y', 'age', 'default', 'housing', 'loan', 'day', 'month', 'campaign','pdays', 'previous', 'marital_divorced', 'marital_single',
                'poutcome_unknown', 'poutcome_failure', 'poutcome_other', 'education_secondary', 'education_tertiary', 'contact_telephone',
                'contact_cellular', 'job_technician', 'job_student', 'job_unemployed', 'job_blue-collar', 'job_management', 'job_services', 'job_admin.',
                'job_retired', 'job_entrepreneur', 'job_housemaid', 'balance_over avg', 'balance_avg']

  # reindex
  dtf = dtf.reindex(columns=cols_order, fill_value=0)

  # Return
  return dtf.drop('y', axis=1), dtf['y']

# Function to predict a single entry
def predict_single_entry_all_columns(observation, model):
  '''
  This function takes in a list and returns a prediction whether the customer will or not submit a term direct deposit.
  * Input:
  - observation: dictionary
  * Output:
  - prediction probability: 0 or 1
  '''
  # dictionary to dataframe
  dfp = pd.DataFrame(observation, index=[0])

  # Prepare Data
  predictors, target = prepare_data_all_columns(dfp)
  
  # Predict
  test_prediction = model.predict_proba(predictors)

  # Return result
  return test_prediction


# Function to prepare the data for prediction
def prepare_data_simpler_model(dtf):
  '''
  Function to prepare the data for prediction. The steps are the same used to prepare the training data.
  * Input:
  dtf: dataframe

  * Output:
  data clean: dataframe with the cleaned data
  '''

  # Imports
  from feature_engine.encoding import OneHotEncoder
  from feature_engine.imputation import CategoricalImputer
  import category_encoders as ce


  # Simple Imputer for education
  imputer = CategoricalImputer(
      imputation_method="frequent"  )

  # Fit and Transform
  imputer.fit(dtf)
  dtf = imputer.transform(dtf)

  # Fill NAs with "unknown" in contact
  dtf['contact'] = dtf['contact'].fillna('unknown')

  # Binarizing default, housing, loan, and y
  dtf = dtf.replace({'no': 0, 'yes': 1})

  # Instance OHR
  ohe = OneHotEncoder(variables=['contact'], drop_last=True)

  # Fit
  ohe.fit(dtf)

  # Transform
  dtf = ohe.transform(dtf)

  # Move y to the first column
  dtf.insert(0, 'y', dtf.pop('y'))

  # Month to numbers
  dtf['month'] = dtf['month'].map({ 'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6, 'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12})

  # drop variable duration
  # dtf = dtf.drop('duration', axis=1)

  # Transforming variable Age into bins
  # Using Sturges rule, where number of bins k = 1 + 3.3*log10(n)
  k = int( 1 + 3.3*np.log10(len(dtf)) )

  # Categorize age, balance, duration, previous, pdays
  
  dtf['pdays'] = variable_to_category(dtf, 'pdays', k=k)

  # CatBoost Encoding the dataset
  dtf = ce.CatBoostEncoder().fit_transform(dtf, dtf['y'])

  # Reindex to match columns from the fitted model
  # cols_order = ['y','default', 'housing', 'loan', 'day', 'contact_cellular', 'contact_telephone', 'month', 'campaign', 'pdays']

  cols_order = ['default', 'housing', 'loan', 'day', 'contact_cellular', 'contact_telephone', 'month', 'campaign', 'pdays','y']

  # reindex
  dtf = dtf.reindex(columns=cols_order, fill_value=0)

  # Return
  return dtf.drop('y', axis=1), dtf['y']


# Function to prepare the data for prediction
def prepare_data_simpler_streamlit(dtf):
  '''
  Function to prepare the data for prediction. The steps are the same used to prepare the training data.
  * Input:
  dtf: dataframe

  * Output:
  data clean: dataframe with the cleaned data
  '''

  # Imports
  from feature_engine.encoding import OneHotEncoder
  from feature_engine.imputation import CategoricalImputer
  import category_encoders as ce


  # Simple Imputer for education
  imputer = CategoricalImputer(
      imputation_method="frequent"  )

  # Fit and Transform
  imputer.fit(dtf)
  dtf = imputer.transform(dtf)

  # Binarizing default, housing, loan, and y
  dtf = dtf.replace({'no': 0, 'yes': 1})

  # Fit
  # ohe.fit(dtf)

  # # Transform
  # dtf = ohe.transform(dtf)

  # Move y to the first column
  dtf.insert(0, 'y', dtf.pop('y'))

  # Month to numbers
  dtf['month'] = dtf['month'].map({ 'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6, 'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12})

  # drop variable duration
  # dtf = dtf.drop('duration', axis=1)

  # Transforming variable Age into bins
  # Using Sturges rule, where number of bins k = 1 + 3.3*log10(n)
  k = int( 1 + 3.3*np.log10(len(dtf)) )

  # Categorize age, balance, duration, previous, pdays
  
  dtf['pdays'] = variable_to_category(dtf, 'pdays', k=k)

  # CatBoost Encoding the dataset
  dtf = ce.CatBoostEncoder().fit_transform(dtf, dtf['y'])

  # Reindex to match columns from the fitted model
  # cols_order = ['y','default', 'housing', 'loan', 'day', 'contact_cellular', 'contact_telephone', 'month', 'campaign', 'pdays']

  cols_order = ['default', 'housing', 'loan', 'day', 'contact_cellular', 'contact_telephone', 'month', 'campaign', 'pdays','y']

  # reindex
  dtf = dtf.reindex(columns=cols_order, fill_value=0)

  # Return
  return dtf.drop('y', axis=1)



# Function to predict a single entry
def predict_single_entry_simpler(observation, model):
  '''
  This function takes in a list and returns a prediction whether the customer will or not submit a term direct deposit.
  * Input:
  - observation: dictionary
  * Output:
  - prediction probability: 0 or 1
  '''
  # dictionary to dataframe
  dfp = pd.DataFrame(observation, index=[0])

  # Prepare Data
  predictors, target = prepare_data_simpler_model(dfp)
  
  # Predict
  test_prediction = model.predict_proba(predictors)

  # Return result
  return test_prediction


