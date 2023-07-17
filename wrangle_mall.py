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
     
    conn = get_db_url('mall_customers')

    query = '''
           SELECT *
            FROM customers;
            '''

    
    df = pd.read_sql(query, conn)
    return df

def get_mall_data():
   
    if os.path.isfile('mall_df.csv'):
        df = pd.read_csv('mall_df.csv', index_col = 0)
        

    else:

        df = new_mall_data()
        df.to_csv('mall_df.csv')
        
    return df

# -----------------------------outliers--------------------------------

def detect_outliers_iqr(df, columns, threshold=1.5):
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
# -----------------------------prep--------------------------------
def prep_mall_data(df):
    
    
    def handle_missing_values(df, prop_required_column=0.5, prop_required_row=0.5):
       
        threshold_col = int(round(prop_required_column * len(df.index)))
        df = df.dropna(thresh=threshold_col, axis=1)
    
        threshold_row = int(round(prop_required_row * len(df.columns)))
        df = df.dropna(thresh=threshold_row)
    
        return df

    df = handle_missing_values(df)
    
    encoded_cols = pd.get_dummies(df['gender'], prefix='gender')
    df = pd.concat([df, encoded_cols], axis=1)
    
    train, validate, test = split_mall_data(df)
    
    return train, validate, test
# -----------------------------split--------------------------------



def split_mall_data(df):
    
  
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, 
                                       test_size=.3, 
                                       random_state=123) 
                                       

    
    return train, validate, test
    
# -----------------------------scale--------------------------------
    
def scale_data(train, validate, test):
    
    
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


def wrangle_mall():
  
    df = get_mall_data()
    train, validate, test = prep_mall_data(df)
    return train, validate, test