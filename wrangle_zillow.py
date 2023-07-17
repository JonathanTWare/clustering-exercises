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
           SELECT p.bedroomcnt, p.bathroomcnt, p.calculatedfinishedsquarefeet,p.fips, p.yearbuilt, p.taxvaluedollarcnt
            FROM properties_2017 p
            JOIN predictions_2017 pr ON p.parcelid = pr.parcelid
            WHERE p.propertylandusetypeid = 261;
            '''

    
    df = pd.read_sql(query, conn)
    return df




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
