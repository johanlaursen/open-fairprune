import numpy as np
import pandas as pd
import seaborn as sns

df = pd.read_csv("data/Train_Dataset.csv")

print(df["Default"].value_counts())

# Null values in each column and percentage missing:
print(df.isnull().sum().sort_values(ascending=False))
print((df.isnull().sum() / len(df)).sort_values(ascending=False))


# Clean Client_Income column
df["Client_Income"] = df["Client_Income"].str.replace("$", "").replace("", None).astype(float)

# logarithmic histplot of Client_Income
sns.histplot(df["Client_Income"], log_scale=True, bins=15)

print(df["Social_Circle_Default"].value_counts())

print(df["Client_Occupation"].value_counts())
