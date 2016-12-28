import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
df1=pd.read_csv('C:/Users/chaithu/Desktop/Anomization/train_flow.csv')
X=df1[col_names1]
y=df1['class1']
model = ExtraTreesClassifier()
model.fit(X,y)
indices = np.argsort(model.feature_importances_)[::-1]
plt.figure(figsize=(10 * 2, 10))
index = np.arange(len(col_names1))
bar_width = 0.35
plt.bar(index, model.feature_importances_*5, color='red', alpha=0.5)
plt.xlabel('features')
plt.ylabel('importance')
plt.title('Feature importance')
plt.xticks(index + bar_width,range(len(col_names1)))
plt.tight_layout()
plt.show()
g=model.feature_importances_*5
h=pd.DataFrame(g)
l=pd.DataFrame(col_names1)
l[1]=h.values
l.columns=['Feature','INFO_GAIN']
print(l)
