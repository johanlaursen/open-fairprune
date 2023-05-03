import datawig
import pandas as pd
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import StandardScaler

from open_fairprune.data_util import FLOAT_COLUMNS, get_split_df


def main():
    train_df = get_split_df("train")

    # Null values in each column and percentage missing:
    print((train_df.isnull().sum() / len(train_df)).sort_values(ascending=False))

    # Dropping everything with more than 50% missing values and score source 3
    train_df = train_df.dropna(thresh=len(train_df) / 2, axis=1)
    # df = df.drop(['Score_Source_3'], axis=1)

    # Replacing nan values in Client Occupation with 'nan'
    train_df["Client_Occupation"] = train_df["Client_Occupation"].fillna("nan")
    print(train_df["Client_Occupation"].value_counts())

    # logarithmic histplot of Client_Income
    # sns.histplot(train_df['Client_Income'], log_scale=True, bins=15)

    y_train = train_df["Default"]
    X_train = train_df.drop(["Default"], axis=1)

    # Standardize float columns
    X_train[FLOAT_COLUMNS] = StandardScaler().fit_transform(X_train[FLOAT_COLUMNS])

    # Write down mse to a file
    with open("mse.txt", "w") as file:
        for i, col in enumerate(FLOAT_COLUMNS):
            numerical_imputer = datawig.SimpleImputer(
                input_columns=FLOAT_COLUMNS,  # column(s) containing information about the column we want to impute
                output_column=col,  # the column we'd like to impute values for
                output_path=f"{col}_imputer_model",  # stores model data and metrics
            )

            numerical_imputer.fit(train_df=X_train)

            imputed = numerical_imputer.predict(X_train)

            # Get rows where Credit Bureau is not null
            not_null = imputed[imputed[col].notnull()]

            male_df = not_null[not_null["Client_Gender"] == "Male"]
            female_df = not_null[not_null["Client_Gender"] == "Female"]

            male_mse = mse(male_df[f"{col}_imputed"], male_df[col])
            female_mse = mse(female_df[f"{col}_imputed"], female_df[col])
            print(f"Finished column {i}/{len(FLOAT_COLUMNS)-1}")
            file.write(f"{col}:\nMale RMSE: {male_mse}\nFemale RMSE: {female_mse}\n")


if __name__ == "__main__":
    main()

"""categorical_columns = ['Accompany_Client', 'Client_Income_Type', 'Client_Education', 'Client_Marital_Status', 'Client_Gender', 'Loan_Contract_Type', 'Client_Housing_Type', 'Client_Occupation', 'Client_Permanent_Match_Tag', 'Client_Contact_Work_Tag', 'Type_Organization']

categorical_imputer = datawig.SimpleImputer(
    input_columns=categorical_columns, # column(s) containing information about the column we want to impute
    output_column='Loan_Contract_Type', # the column we'd like to impute values for
    output_path = 'categorical_imputer_model' # stores model data and metrics
    )

categorical_imputer.fit(train_df=X_train)

categorical_imputed = categorical_imputer.predict(X_train)
"""
