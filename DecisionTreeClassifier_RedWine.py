import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
data=pd.read_csv('Wine.csv',header=None)
#欄位0（Target）：紅酒的分類（總共分為3類，分別為1~3） 
# Column 0 (Target): The classification of wines (divided into 3 classes: 1, 2, 3)
#欄位1-13（Data）：各種紅酒中各項化學成分檢驗結果，包含如：酒精、蘋果酸、鎂、黃酮、顏色強度、色澤…等等。
# Columns 1-13 (Data): Various chemical composition test results of different wines, including alcohol, malic acid, magnesium, flavonoids, color intensity, hue, etc.
x=data.iloc[:,1:]
y=data.iloc[:,0]
#建立分類器
# Create the classifier
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.25,random_state=5)
model=DecisionTreeClassifier()
model.fit(train_x,train_y)
# 預測結果並評估模型的準確度
# Predict results and evaluate model accuracy
pred_y=model.predict(test_x)
from sklearn.metrics import accuracy_score
print(f'決策數分類器的準確度為:{round(accuracy_score(test_y,pred_y)*100,2)}')
print(f' Decision tree classifier accuracy: {round(accuracy_score(test_y, pred_y) * 100, 2)}')
#預測資料:[1.51, 1.73, 1.98, 20.15, 85, 2.2, 1.92, .32, 1.48, 2.94, 1, 3.57, 172]
# Prediction data: [1.51, 1.73, 1.98, 20.15, 85, 2.2, 1.92, 0.32, 1.48, 2.94, 1, 3.57, 172]
to_be_predicted=np.array([1.51, 1.73, 1.98, 20.15, 85, 2.2, 1.92, .32, 1.48, 2.94, 1, 3.57, 172]).reshape(1,-1)
to_be_predicted=pd.DataFrame(to_be_predicted,columns=x.columns)
print(f'預測結果為第{model.predict(to_be_predicted)[0]}類')
print(f' Predicted class is: {model.predict(to_be_predicted)[0]}')
#預測資料:[14.23, 1.71, 2.43, 15.6, 127, 2.8, 3.06, .28, 2.29, 5.64, 1.04, 3.92,1065]
# Prediction data: [14.23, 1.71, 2.43, 15.6, 127, 2.8, 3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065]
to_be_predicted=np.array([14.23, 1.71, 2.43, 15.6, 127, 2.8, 3.06, .28, 2.29, 5.64, 1.04, 3.92,1065]).reshape(1,-1)
to_be_predicted=pd.DataFrame(to_be_predicted,columns=x.columns)
print(f'預測結果為第{model.predict(to_be_predicted)[0]}類')
print(f' Predicted class is: {model.predict(to_be_predicted)[0]}')
#預測資料:[13.71, 5.65, 2.45, 20.5, 95, 1.68, .61, .52, 1.06, 7.7, .64, 1.74, 720]
# Prediction data: [13.71, 5.65, 2.45, 20.5, 95, 1.68, 0.61, 0.52, 1.06, 7.7, 0.64, 1.74, 720]
to_be_predicted=np.array([13.71, 5.65, 2.45, 20.5, 95, 1.68, .61, .52, 1.06, 7.7, .64, 1.74, 720]).reshape(1,-1)
to_be_predicted=pd.DataFrame(to_be_predicted,columns=x.columns)
print(f'預測結果為第{model.predict(to_be_predicted)[0]}類')
print(f' Predicted class is: {model.predict(to_be_predicted)[0]}')
