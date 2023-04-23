import pandas as pd
from sklearn import metrics
import os
from sklearn.linear_model import LogisticRegression
#from . import dispatchers
import joblib
#from . import categorical_variables

def predict(test_data_path, model_type, model_path):
    df = pd.read_csv(test_data_path)
    test_id = df['id'].values
    df.drop(['id'],axis=1,inplace=True)
    predictions = None
    print(df.shape)
    target_columns = ["['prognosis']_Chikungunya", "['prognosis']_Dengue", "['prognosis']_Japanese_encephalitis", 
                      "['prognosis']_Lyme_disease", "['prognosis']_Malaria", "['prognosis']_Plague", 
                      "['prognosis']_Rift_Valley_fever", "['prognosis']_Tungiasis", 
                      "['prognosis']_West_Nile_fever", "['prognosis']_Yellow_Fever", "['prognosis']_Zika"]

    for FOLD in range(5):
        final_results = pd.DataFrame()
        confidence = pd.DataFrame()
        for classes in target_columns:
            df = pd.read_csv(test_data_path)
            clf = joblib.load(os.path.join(model_path, f"{model_type}_{FOLD}_{classes}.pkl"))
            
            preds = clf.predict_proba(df.iloc[:,0:64])[:,0]
            confidence[classes] = clf.predict_proba(df.iloc[:,0:64])[:,1]
            final_results[classes] = preds


        final_results.to_csv(f"final_results_{FOLD}.csv")
        confidence.to_csv(f"confidence_{FOLD}.csv")
        
    

if __name__ == "__main__":
    result = predict(test_data_path='input/test.csv',model_type='logisticsClassifier',model_path='models/')
    print(result)

    


