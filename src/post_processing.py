import pandas as pd
import numpy as np

## fold 0 see max probability across all classes for all rows

final_results_0 = pd.read_csv("final_results_0.csv")
final_results_1 = pd.read_csv("final_results_1.csv")
final_results_2 = pd.read_csv("final_results_2.csv")
final_results_3 = pd.read_csv("final_results_3.csv")
final_results_4 = pd.read_csv("final_results_4.csv")

result_1 = final_results_0.add(final_results_1)
result_2 = result_1.add(final_results_2)
result_3 = result_2.add(final_results_3)
result_4 = result_3.add(final_results_4)

predictions = result_4/5

predictions.drop(['Unnamed: 0'],axis=1,inplace=True)

predictions.to_excel("predictions.xlsx")




