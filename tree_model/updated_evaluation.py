# import packages
import pandas as pd
import score
from sklearn.metrics import f1_score

# read the data
### predicted y
predict_df = pd.read_csv("competition_test_stances.csv", encoding='iso-8859-1')
### true y
origin_df = pd.read_csv("tree_pred_cor2.csv", encoding = 'iso-8859-1')

# confusion matrix and official score
score.report_score(origin_df['Stance'], predict_df['Stance'])

# f1_score original
macro_score = f1_score(origin_df['Stance'], predict_df['Stance'], average='macro')
print("Macro score: " + str(macro_score))
# f1_score weighted
weighted_score = f1_score(origin_df['Stance'], predict_df['Stance'], average='weighted')
print("Weighted score: " + str(weighted_score))
