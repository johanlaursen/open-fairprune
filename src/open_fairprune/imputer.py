from datawig import SimpleImputer
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

    # Remove random symbols in the data
    def isfloat(x):
        try:
            float(x)
            return True
        except:
            return False
        
    float_columns = ['Client_Income', 'Credit_Amount', 'Loan_Annuity', 'Population_Region_Relative', 'Score_Source_2', 'Car_Owned', 'Bike_Owned', 'Active_Loan', 'House_Own', 'Child_Count', 'Age_Days', 'Employed_Days', 'Registration_Days', 'ID_Days', 'Client_Family_Members', 'Cleint_City_Rating', 'Application_Process_Day', 'Application_Process_Hour', 'Phone_Change', 'Credit_Bureau']
    train_df.loc[:,float_columns] = train_df[train_df[float_columns].applymap(isfloat)].astype(float)

    #Dropping everything with more than 50% missing values and score source 3
    train_df = train_df.dropna(thresh=len(train_df)/2, axis=1)

    #Replacing nan values in Client Occupation with 'nan'
    train_df['Client_Occupation'] = train_df['Client_Occupation'].fillna('nan')

    y_train = train_df['Default']
    X_train = train_df.drop(['Default'], axis=1)

    #Standardize float columns
    X_train[float_columns] = StandardScaler().fit_transform(X_train[float_columns])

    for i, col in enumerate(float_columns):

        imputer = SimpleImputer.load(f'../../data/{col}_imputer_model')
        imputed = imputer.predict(X_train)
        temp_Series = X_train[col]
        X_train[col] = temp_Series.fillna(imputed[f'{col}_imputed'])
        print(f'Finished: {i}/{len(float_columns)-1}')

    X_train.to_csv(path_or_buf='Train_Dataset_Imputed.csv')

if __name__ == "__main__":
    main()