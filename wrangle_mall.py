import pandas as pd
import numpy as np
from env import get_db_url
import os
from sklearn.model_selection import train_test_split
import pandas as pd
from pydataset import data
from env import get_db_url
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# -----------------------------acquire--------------------------------


def new_mall_data():
    """
    Fetches mall data from the database using SQL query.

    Returns:
    - df: DataFrame containing mall data
    """
   
    conn = get_db_url('mall_customers')

    query = '''
           SELECT *
            FROM customers;
            '''

    
    df = pd.read_sql(query, conn)
    return df

# -----------------------------outliers--------------------------------

def detect_outliers_iqr(df, columns, threshold=1.5):
    """
    Detect outliers in the specified numeric column(s) of a DataFrame using the IQR method.
    
    Args:
        df (pandas.DataFrame): The DataFrame to detect outliers.
        columns (list or str): The column name(s) or list of column names to examine for outliers.
        threshold (float, optional): The threshold value to determine outliers (default is 1.5).
        
    Returns:
        pandas.DataFrame: A DataFrame containing the outliers for the specified column(s).
    """
    if isinstance(columns, str):
        columns = [columns]
    
    outliers = pd.DataFrame()
    
    for col in columns:
        # Calculate the IQR
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        
        # Determine the outlier thresholds
        lower_threshold = q1 - threshold * iqr
        upper_threshold = q3 + threshold * iqr
        
        # Identify outliers
        outliers_col = df[(df[col] < lower_threshold) | (df[col] > upper_threshold)]
        outliers = pd.concat([outliers, outliers_col], ignore_index=True)
    
    return outliers

# -----------------------------split--------------------------------

def split_zillow_data(df):
    """
    Splits the Mall data into training, validation, and test sets.

    Arguments:
    - df: DataFrame containing Mall data

    Returns:
    - train: DataFrame of the training set
    - validate: DataFrame of the validation set
    - test: DataFrame of the test set
    """
  
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, 
                                       test_size=.3, 
                                       random_state=123) 
                                       

    
    return train, validate, test

# -----------------------------missingval--------------------------------

def handle_missing_values(df, prop_required_column=0.5, prop_required_row=0.5):
        """
        Drops columns and rows from a DataFrame based on the percentage of missing values.

        Args:
            df (pandas.DataFrame): The DataFrame to handle missing values.
            prop_required_column (float, optional): The minimum proportion of non-null values required for columns (default is 0.5).
            prop_required_row (float, optional): The minimum proportion of non-null values required for rows (default is 0.5).

        Returns:
            pandas.DataFrame: The DataFrame with dropped rows and columns.
        """

        threshold_col = int(round(prop_required_column * len(df.index)))
        df = df.dropna(thresh=threshold_col, axis=1)


        threshold_row = int(round(prop_required_row * len(df.columns)))
        df = df.dropna(thresh=threshold_row)

        return df
    
# -----------------------------scale--------------------------------
    
def scale_data(train, validate, test):
    """
        Scales numerical columns in the train, validate, and test sets using MinMaxScaler.

        Arguments:
        - train: DataFrame of the training set
         - validate: DataFrame of the validation set
        - test: DataFrame of the test set

        Returns:
        - train_scaled: Scaled training set DataFrame
         - validate_scaled: Scaled validation set DataFrame
        - test_scaled: Scaled test set DataFrame
         """
    
    numeric_cols = ['spending_score','annual_income','age']
    
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    
    scaler = MinMaxScaler()
    scaler.fit(train[numeric_cols])
    
    train_scaled[numeric_cols] = scaler.transform(train[numeric_cols])
    validate_scaled[numeric_cols] = scaler.transform(validate[numeric_cols])
    test_scaled[numeric_cols] = scaler.transform(test[numeric_cols])
    
    return train_scaled, validate_scaled, test_scaled