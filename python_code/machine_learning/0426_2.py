import pandas as pd
import numpy as np

"""
문제 2. 
Pcall를 groupby하여서 Age_max, Age_min, Age_mean, Parch_max, Parch_min, Sibsp_max, Sibsp_min을 구하여라
"""

titanic_df = pd.read_csv("C:/jeon/titanic_train.csv")
titanic_df_my = titanic_df

agg_format = {"Age": ["max", "min", "mean"], "Parch": ["max", "min"], "SibSp": ["max", "min"]}
titanic_df_my = titanic_df_my.groupby(by="Pclass").agg(agg_format)
