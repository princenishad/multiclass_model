import pandas as pd
from sklearn import metrics
import os
from sklearn.linear_model import LogisticRegression
from . import dispatchers
import joblib
from . import categorical_variables
import numpy as np

TRAINING_DATA = os.environ.get("TRAINING_DATA")
FOLD = int(os.environ.get("FOLD"))
MODEL = os.environ.get('MODEL')

Fold_Mappings = {
    0 : [1,2,3,4],
    1 : [0,2,3,4],
    2 : [0,1,3,4],
    3 : [0,1,2,4],
    4 : [0,1,2,3]
}

if __name__ == "__main__":
    df = pd.read_csv("input/train_fold.csv")
    df = categorical_variables.CategoricalFeatures.onehotencoding(df,['prognosis'],'disease')
    train_df = df[df.kfold.isin(Fold_Mappings.get(FOLD))]
    val_df = df[df.kfold==FOLD]

    

    ## the problem is Multiclass now

    target_columns = [c for c in df.columns if "['prognosis']" in c]
    #print(target_columns)
    ytrain = train_df[target_columns]
    yval = val_df[target_columns]

    for features in target_columns:
        train_df.drop([features],axis=1,inplace=True)
        val_df.drop([features],axis=1,inplace=True)
    
    train_df.drop(['kfold','Unnamed: 0','id'],axis=1,inplace=True)
    val_df.drop(['kfold', 'Unnamed: 0','id'],axis=1,inplace=True)
    #print(train_df.shape())
    model_type = 'logisticsClassifier'
    result = pd.DataFrame()
    for classes in target_columns:
        clf = dispatchers.MODELS[model_type]
        clf.fit(train_df,ytrain[classes])
        pred_probability = clf.predict_proba(val_df)[:,0:1]
        result[classes] = list(clf.predict_proba(val_df))
        print("AUC-ROC score is :", metrics.roc_auc_score(yval,pred_probability))
        joblib.dump(clf, f"models/{model_type}_{FOLD}_{classes}.pkl")

    result.to_csv(f"models/{FOLD}_fold_predictions.csv")






    


