import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# from open_fairprune.data_util import LoanDataset

FLOAT_COLUMNS = [
    "Active_Loan",
    "Age_Days",
    "Application_Process_Day",
    "Application_Process_Hour",
    "Bike_Owned",
    "Car_Owned",
    "Child_Count",
    "Cleint_City_Rating",
    "Client_Family_Members",
    "Client_Income",
    "Credit_Amount",
    "Credit_Bureau",
    "Default",
    "Employed_Days",
    "Homephone_Tag",
    "House_Own",
    "ID",
    "ID_Days",
    "Loan_Annuity",
    "Mobile_Tag",
    "Own_House_Age",
    "Phone_Change",
    "Population_Region_Relative",
    "Registration_Days",
    "Score_Source_1",
    "Score_Source_2",
    "Score_Source_3",
    "Social_Circle_Default",
    "Workphone_Working",
]


def main():
    df = pd.read_csv("Train_Dataset.csv")
    splits = {
        "train": df.ID % 7 <= 4,
        "dev": df.ID % 7 == 5,
        "test": df.ID % 7 == 6,  # Around 15%
    }
    df = df[splits["train"]]

    def isfloat(x):
        try:
            float(x)
            return True
        except:
            return False

    df.loc[:, FLOAT_COLUMNS] = df[FLOAT_COLUMNS][df[FLOAT_COLUMNS].applymap(isfloat)]
    train_df = df.astype({c: "float" for c in FLOAT_COLUMNS})

    train_df["Application_Process_Hour"].value_counts()

    # Null values in each column and percentage missing:
    print((train_df.isnull().sum() / len(train_df)).sort_values(ascending=False))

    # Dropping everything with more than 50% missing values except score sources
    train_df = train_df.drop(["Own_House_Age", "Social_Circle_Default"], axis=1)

    # Replacing nan values in Client Occupation with 'Unknown'
    train_df["Client_Occupation"] = train_df["Client_Occupation"].fillna("Unknown")
    print(train_df["Client_Occupation"].value_counts())

    # Replacing ## with nan values in Accompany Client
    train_df["Accompany_Client"] = train_df["Accompany_Client"].replace("##", np.nan)

    # logarithmic histplot of Client_Income
    # sns.histplot(train_df['Client_Income'], log_scale=True, bins=15)


if __name__ == "__main__":
    main()
