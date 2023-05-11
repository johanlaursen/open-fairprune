import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import StandardScaler

def main():
    df = pd.read_csv('../../data/Train_Dataset.csv')

    splits = {
        "train": df.ID % 7 <= 4,
        "dev": df.ID % 7 == 5,
        "test": df.ID % 7 == 6 #Around 15%
    }
    train_df = df[splits["train"]].copy()

    # Dropping everything with more than 50% missing values except score sources
    train_df = train_df.drop(["Own_House_Age", "Social_Circle_Default"], axis=1)

    # Remove random symbols in the data
    def isfloat(x):
        try:
            float(x)
            return True
        except:
            return False
        
    float_columns = ['Client_Income', 'Credit_Amount', 'Loan_Annuity', 'Population_Region_Relative', 'Score_Source_1', 'Score_Source_2', 'Score_Source_3', 'Car_Owned', 'Bike_Owned', 'Active_Loan', 'House_Own', 'Child_Count', 'Employed_Days', 'Registration_Days', 'ID_Days', 'Client_Family_Members', 'Cleint_City_Rating', 'Application_Process_Day', 'Application_Process_Hour', 'Phone_Change', 'Credit_Bureau', 'Age_Days']
    train_df.loc[:,float_columns] = train_df[train_df[float_columns].applymap(isfloat)].astype(float)

    train_df = train_df.dropna(subset=["Age_Days"])  # NOTE: Drops where we dont have label!
    train_df["Age_Days"] = train_df["Age_Days"].astype(int) // 365 > 43 # Median Age
    float_columns.pop() # Remove Age_Days from floats since it's now a boolean

    #Dropping everything with more than 50% missing values
    #train_df = train_df.dropna(thresh=len(train_df)/2, axis=1)

    #Replacing nan values in Client Occupation with 'nan'
    train_df['Client_Occupation'] = train_df['Client_Occupation'].fillna('nan')

    y_train = train_df['Default']
    X_train = train_df.drop(['Default'], axis=1)

    #Standardize float columns
    X_train[float_columns] = StandardScaler().fit_transform(X_train[float_columns])

    X_train['Accompany_Client'] = X_train['Accompany_Client'].replace('##', np.nan)

    for i, col in enumerate(float_columns):

        imputer = SimpleImputer.load(f'../../data/{col}_imputer_model')
        imputed = imputer.predict(train_df)
        temp_Series = train_df[col]
        train_df[col] = temp_Series.fillna(imputed[f'{col}_imputed'])
        print(f'Finished: {i}/{len(float_columns)-1} numerical columns')

    train_df.to_csv(path_or_buf='Train_Dataset_Imputed.csv')

if __name__ == "__main__":
    main()