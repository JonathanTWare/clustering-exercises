#import libraries
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


def new_zillow_data():
    """
    Fetches Zillow data from the database using SQL query.

    Returns:
    - df: DataFrame containing Zillow data
    """
   
    conn = get_db_url('zillow')

    query = '''
        SELECT *
    FROM properties_2017 AS p
    JOIN predictions_2017 AS pr ON p.parcelid = pr.parcelid
    LEFT JOIN airconditioningtype AS act ON p.airconditioningtypeid = act.airconditioningtypeid
    LEFT JOIN architecturalstyletype AS ast ON p.architecturalstyletypeid = ast.architecturalstyletypeid
    LEFT JOIN buildingclasstype AS bct ON p.buildingclasstypeid = bct.buildingclasstypeid
    LEFT JOIN heatingorsystemtype AS hot ON p.heatingorsystemtypeid = hot.heatingorsystemtypeid
    LEFT JOIN propertylandusetype AS plt ON p.propertylandusetypeid = plt.propertylandusetypeid
    LEFT JOIN storytype AS st ON p.storytypeid = st.storytypeid
    LEFT JOIN typeconstructiontype AS tct ON p.typeconstructiontypeid = tct.typeconstructiontypeid
    LEFT JOIN unique_properties AS up ON p.parcelid = up.parcelid
    WHERE p.unitcnt = 1;
    
    '''

    
    df = pd.read_sql(query, conn)
    return df


def get_zillow_data():
   
    if os.path.isfile('zillow_df.csv'):
        df = pd.read_csv('zillow_df.csv', index_col = 0)
        

    else:

        df = new_zillow_data()
        df.to_csv('zillow_df.csv')
        
    return df



def prep_zillow_data(df):
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
    
    def drop_duplicate_columns(df):
        """
        Drops duplicate columns from a DataFrame.

        Args:
            df (pandas.DataFrame): The DataFrame to drop duplicate columns from.

        Returns:
            pandas.DataFrame: The DataFrame with duplicate columns removed.
        """
        # Transpose the DataFrame to consider columns as rows for duplicate checking
        transposed_df = df.T

        # Drop duplicate columns
        deduplicated_df = transposed_df.drop_duplicates()

        # Transpose the DataFrame back to the original shape
        df = deduplicated_df.T

        return df


    return df
#-------------------------------------------

def split_zillow_data(df):
    
  
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, 
                                       test_size=.3, 
                                       random_state=123) 
                                       

    
    return train, validate, test

#----------------------------------------------------

def scale_data(train, validate, test):
    
    
    numeric_cols = ['bathroomcnt','taxamount','yearbuilt']
    
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    
    scaler = MinMaxScaler()
    scaler.fit(train[numeric_cols])
    
    train_scaled[numeric_cols] = scaler.transform(train[numeric_cols])
    validate_scaled[numeric_cols] = scaler.transform(validate[numeric_cols])
    test_scaled[numeric_cols] = scaler.transform(test[numeric_cols])
    
    return train_scaled, validate_scaled, test_scaled


    
    
    
    
    
    
    
    
def get_most_recent_transactions(df):
    
    duplicates_df = df[df.duplicated(subset='parcelid', keep=False)]

    
    sorted_df = duplicates_df.sort_values(by='transactiondate', ascending=False)

    
    most_recent_transactions = sorted_df.drop_duplicates(subset='parcelid', keep='first')

    return most_recent_transactions


def check_duplicates(df):
    """
    Checks for duplicates in a DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame to check for duplicates.

    Returns:
        bool: True if duplicates are found, False otherwise.
    """
    duplicates = df.duplicated('parcelid')
    if duplicates.any():
        return True
    else:
        return False
    
    
def wrangle_zillow():
    df = get_zillow_data()
    df = prep_zillow_data(df)
    train, validate, test = split_zillow_data(df)
    train_scaled, validate_scaled, test_scaled = scale_data(train, validate, test)
    return train_scaled, validate_scaled, test_scaled
    



