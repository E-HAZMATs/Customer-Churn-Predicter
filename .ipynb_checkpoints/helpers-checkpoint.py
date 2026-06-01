import pandas as pd

def encode_data(df):
    '''
        Encodes the IBM churn dataset and substitute for missing values returns a new dataset.
    '''
    df = df.copy()
    
    # Getting indices of cols whose value is either Yes or No
    yes_no_cols = [col for col in df.columns if set(df[col].dropna().unique()) == {'Yes', 'No'}]
    df[yes_no_cols] = df[yes_no_cols].replace({'Yes': 1, 'No': 0})

    df['gender'] = df['gender'].replace({'Male': 1, "Female": 0})


    # 3 discrete string values.
    # CHECK: This is bad (same point as df['MultipleLines'] encoding below)?
    yes_no_noInternet_cols = [col for col in df.columns 
            if set(df[col].dropna().unique()) == {'Yes', 'No', 'No internet service'}]
    df[yes_no_noInternet_cols] = df[yes_no_noInternet_cols].replace({"No": 0, "Yes": 1, 'No internet service': 2}).astype(int)
    
    
    # Features whose values are either 0 or 1 but are dtype = object. Need to cast them to int.
    obj_binary_cols = [col for col in df.columns 
                       if df[col].dtype == 'object' 
                       and set(df[col].dropna().unique()) == {0, 1}]
    df[obj_binary_cols] = df[obj_binary_cols].astype(int)

    # Customers have empty TotalCharges entry. Can't cast to float till I give them value.
    # I assigned them a value of their "tenure" * "MonthlyCharges". Other customer's totCharge is close to that.

    # Getting indices of all customers whose totalCharges = ' '.
    mask = df['TotalCharges'].str.strip() == ''
    # Assign value as str since the column was already string. We cast to float after.
    df.loc[mask, 'TotalCharges'] = (df.loc[mask, 'tenure'] * df.loc[mask, 'MonthlyCharges']).astype(str)

    # Casting column to be float.
    df['TotalCharges'] = df['TotalCharges'].astype(float)

    #CHECK: This is bad? No phone service should be -1?
    df['MultipleLines'] = df['MultipleLines'].replace({"No": 0, 'Yes': 1, 'No phone service': 2}).astype(int)

    # One-hot encode these columns cus there is no a sort of hierarchy between them.
    df = pd.get_dummies(df, columns=['InternetService', 'PaymentMethod'], dtype=int)

    df['Contract'] = df['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})
    
    return df