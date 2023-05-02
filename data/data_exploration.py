import datawig
import pandas as pd
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import StandardScaler

def main():
    df = pd.read_csv('Train_Dataset.csv')

    splits = {
        "train": df.ID % 7 <= 4,
        "dev": df.ID % 7 == 5,
        "test": df.ID % 7 == 6 #Around 15%
    }
    train_df = df[splits["train"]].copy()
    print(train_df['Default'].value_counts())

    # Remove random symbols in the data
    def isfloat(x):
        try:
            float(x)
            return True
        except:
            return False
        
    float_columns = ['Client_Income', 'Credit_Amount', 'Loan_Annuity', 'Population_Region_Relative', 'Score_Source_2', 'Car_Owned', 'Bike_Owned', 'Active_Loan', 'House_Own', 'Child_Count', 'Age_Days', 'Employed_Days', 'Registration_Days', 'ID_Days', 'Client_Family_Members', 'Cleint_City_Rating', 'Application_Process_Day', 'Application_Process_Hour', 'Phone_Change', 'Credit_Bureau']
    train_df.loc[:,float_columns] = train_df[train_df[float_columns].applymap(isfloat)].astype(float)

    # Null values in each column and percentage missing:
    print((train_df.isnull().sum()/len(train_df)).sort_values(ascending=False))

    #Dropping everything with more than 50% missing values and score source 3
    train_df = train_df.dropna(thresh=len(train_df)/2, axis=1)
    #df = df.drop(['Score_Source_3'], axis=1)

    #Replacing nan values in Client Occupation with 'nan'
    train_df['Client_Occupation'] = train_df['Client_Occupation'].fillna('nan')
    print(train_df['Client_Occupation'].value_counts())

    #logarithmic histplot of Client_Income
    #sns.histplot(train_df['Client_Income'], log_scale=True, bins=15)

    y_train = train_df['Default']
    X_train = train_df.drop(['Default'], axis=1)

    #Standardize float columns
    X_train[float_columns] = StandardScaler().fit_transform(X_train[float_columns])

    #Write down mse to a file
    with open('mse.txt', 'w') as file:
        for i, col in enumerate(float_columns):

            numerical_imputer = datawig.SimpleImputer(
                input_columns=float_columns, # column(s) containing information about the column we want to impute
                output_column=col, # the column we'd like to impute values for
                output_path = f'{col}_imputer_model' # stores model data and metrics
                )

            numerical_imputer.fit(train_df=X_train)

            imputed = numerical_imputer.predict(X_train)

            # Get rows where Credit Bureau is not null
            not_null = imputed[imputed[col].notnull()]

            male_df = not_null[not_null['Client_Gender'] == 'Male']
            female_df = not_null[not_null['Client_Gender'] == 'Female']

            male_mse = mse(male_df[f'{col}_imputed'], male_df[col])
            female_mse = mse(female_df[f'{col}_imputed'], female_df[col])
            print(f'Finished column {i}/{len(float_columns)-1}')
            file.write(f"{col}:\nMale RMSE: {male_mse}\nFemale RMSE: {female_mse}\n")

if __name__ == "__main__":
    main()

'''categorical_columns = ['Accompany_Client', 'Client_Income_Type', 'Client_Education', 'Client_Marital_Status', 'Client_Gender', 'Loan_Contract_Type', 'Client_Housing_Type', 'Client_Occupation', 'Client_Permanent_Match_Tag', 'Client_Contact_Work_Tag', 'Type_Organization']

categorical_imputer = datawig.SimpleImputer(
    input_columns=categorical_columns, # column(s) containing information about the column we want to impute
    output_column='Loan_Contract_Type', # the column we'd like to impute values for
    output_path = 'categorical_imputer_model' # stores model data and metrics
    )

categorical_imputer.fit(train_df=X_train)

categorical_imputed = categorical_imputer.predict(X_train)
'''