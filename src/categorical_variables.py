from sklearn import preprocessing
import pandas as pd

class CategoricalFeatures:
    def onehotencoding(df : pd.DataFrame,categorical_column : list,prefix_ : str):
        new_df = pd.get_dummies(df,[categorical_column])
        return new_df
    



