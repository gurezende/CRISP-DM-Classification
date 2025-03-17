## Functions for the project CRISP-DM Notebook

import pandas as pd
import numpy as np
from feature_engine.encoding import OneHotEncoder
import category_encoders as ce
import numpy as np
import math
import plotly.graph_objects as go

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
def prepare_data_select_columns(dtf, cols_order):
  '''
  Function to prepare the data for prediction. The steps are the same used to prepare the training data.
  * Input:
  - dtf: dataframe
  - cols_order: list of columns selected for the model

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
  cols_order = cols_order

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


def generate_levels(num_levels, min_radius=0.0, max_radius=1.0):
    """
    Generate evenly spaced radius values (levels) between a minimum and maximum radius.
    Parameters:
    - num_levels (int): The number of levels to generate.
    - min_radius (float): The minimum radius value (default is 0.0).
    - max_radius (float): The maximum radius value (default is 1.0).
    Returns:
    - list: A list of evenly spaced radius values between `min_radius` and `max_radius`.
    """
    return np.linspace(min_radius, max_radius, num_levels).tolist()


def generate_points(levels, total_probs):
    """
    Distribute total probs across levels (interpreted as radii) 
    proportionally to their circumferences, using floats for precision 
    and adjusting for rounding errors.
    Parameters:
    - levels (list of float): Radii of the circles.
    - total_probs (int): Total number of probs to distribute.
    Returns:
    - list of int: Number of probs assigned to each circle.
    """
    # Calculate circumferences
    circumferences = [2 * math.pi * r for r in levels]
    total_circumference = sum(circumferences)
    
    # Distribute probs proportionally using floats
    proportional_points = [(circumference / total_circumference) * total_probs for circumference in circumferences]
    
    # Convert to integers and track the residuals
    points_per_level = [int(p) for p in proportional_points]
    residuals = [p - int(p) for p in proportional_points]
    
    # Adjust for rounding errors to match total_probs
    while sum(points_per_level) < total_probs:
        max_index = residuals.index(max(residuals))
        points_per_level[max_index] += 1
        residuals[max_index] = 0  # Prevent double adjustments

    return points_per_level


def generate_radii_theta(levels, points_per_level, theta_start, theta_end):
    """
    Generate radii and theta values for points to be plotted in a polar scatter plot.
    Parameters:
    - levels (list of float): A list of radius values (one for each level) defining the distance of each level from the origin.
    - points_per_level (list of int): A list indicating the number of points to generate for each corresponding level.
    - theta_start (float): The starting angle (in radians) for the angular range of the points.
    - theta_end (float): The ending angle (in radians) for the angular range of the points.
    Returns:
    - tuple: Two lists:
        - radii_sorted (list of float): The radii of the points, sorted first by angle and then by radius.
        - theta_sorted (list of float): The angular positions of the points, sorted first by angle and then by radius.
    """
    radii = []
    theta = []

    for level, count in zip(levels, points_per_level):
        level_theta_values = np.linspace(theta_start, theta_end, count, endpoint=True)
        radii.extend([level] * count)  # Same radius for all points at this level
        theta.extend(level_theta_values)  # Add theta values for this level

    # Sort radii and theta by angle (theta), then by radius
    radii_theta_sorted = sorted(zip(radii, theta), key=lambda x: (x[1], x[0]))
    radii_sorted, theta_sorted = zip(*radii_theta_sorted)

    return radii_sorted, theta_sorted


def create_parliament_chart(parties, probs, colors, radii_sorted, theta_sorted, marker_size=10):
    """
    Create a Plotly figure to visualize the distribution of parliament probs.
    Parameters:
    - parties (list of str): Names of the political parties.
    - probs (list of int): Number of probs allocated to each party.
    - colors (list of str): Colors associated with each party, corresponding to their visualization.
    - radii_sorted (list of float): Radii values (distances from the origin) for the points, sorted by angle and radius.
    - theta_sorted (list of float): Angular positions (in degrees) for the points, sorted by angle and radius.
    - marker_size (int, optional): Size of the markers representing the probs (default is 10).
    Returns:
    - plotly.graph_objects.Figure: A Plotly figure visualizing the seat distribution.
    """
    fig = go.Figure()
    party_start_idx = 0
    
    for i, party in enumerate(parties):
        party_end_idx = party_start_idx + probs[i]
        
        fig.add_trace(go.Scatterpolar(
            r=radii_sorted[party_start_idx:party_end_idx],
            theta=theta_sorted[party_start_idx:party_end_idx],
            mode='markers',
            marker=dict(
                size=marker_size,
                color=colors[i],
                #line=dict(width=1, color='black')
            ),
            name=party,
            legendgroup=party,
            hovertemplate='Party: ' + party + '<br>probs: ' + str(probs[i]) + '<extra></extra>'
        ))
        
        party_start_idx = party_end_idx

    return fig


def setup_layout(fig, title, subtitle, legend_orientation=None):
    """
    Set up the layout for the Plotly figure, including subtitle and footer.
    
    Parameters:
    - fig: The Plotly figure object to update.
    - title: The title of the figure.
    - subtitle: The subtitle of the figure.
    - footer: The footer of the figure.
    - legend_orientation: (Optional) Orientation of the legend ('h' for horizontal, 'v' for vertical).
    """
    # Define annotation position
    annotation_position = -0.0885 if legend_orientation == 'h' else -0.1015
    
    layout = dict(
        title=title,
        showlegend=True,
        polar=dict(
            radialaxis=dict(showline=False, showticklabels=False, linecolor='#0e1117', gridcolor = "#0e1117"),
            angularaxis=dict(showline=False, showticklabels=False, linecolor='#0e1117', gridcolor = "#0e1117"),
            bgcolor='#0e1117'
        ),
        height=600,
        width=700,
        font=dict(
            family="Poppins, sans-serif",
        ),
        annotations=[
            dict(
                x=annotation_position, y=1.1, xref='paper', yref='paper',
                text=subtitle, showarrow=False,
                font=dict(size=14, color='grey'),
                xanchor='left'
            )
        ]
    )
    
    # Add legend settings if orientation is provided
    if legend_orientation:
        layout['legend'] = dict(
            orientation=legend_orientation,
            yanchor='bottom',
            y=0.1 if legend_orientation == 'h' else 1,
            xanchor='center',
            x=0.5
        )
    
    fig.update_layout(**layout)


