actual =[1,1,0,1,0,1,1,0,1,0,0,1,0,1,1,0]
predicted =[0,1,0,1,1,1,0,0,1,1,0,1,0,1,0,0]

from sklearn.metrics import accuracy_score
value = accuracy_score(actual,predicted)
#print(value)

from sklearn.metrics import confusion_matrix
value1 =confusion_matrix(actual,predicted)
#print(value1)

import pandas as pd
from sklearn.metrics import classification_report
report =pd.DataFrame(classification_report(actual,predicted,output_dict=True))
print(report)