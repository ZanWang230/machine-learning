import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
data= pd.read_csv("NBApoints.csv")
# 使用 LabelEncoder 對球員位置 (Pos) 進行標籤編碼
LabelEncoder_pos=LabelEncoder()
data['Pos']=LabelEncoder_pos.fit_transform(data['Pos'])
# 使用 LabelEncoder 對球隊名稱 (Tm) 進行標籤編碼
LabelEncoder_Tm=LabelEncoder()
data['Tm']=LabelEncoder_Tm.fit_transform(data['Tm'])
x=data[['Pos','Age','Tm']]
y=data['3P']
model=LinearRegression()
# Here, you can observe that the features do not have strong dependencies on each other (correlations are all below 0.1)
corr_matrix = data[['Pos','Age','Tm','3P']].corr() # 這裡可以看出特徵之間沒有依賴關係過強的現象(係數均未大於0.1)
print(corr_matrix['3P']) #這裡可以得知特徵當中跟預測目標最有關係的是球員的位置(Pos),其中關係最小的是隊伍(Tm)

#----------------------------------------------------------
# # 如果想看熱力圖，可啟動這段程式碼
# # If you want to see a heatmap, you can uncomment this section
# import seaborn as sns
# import matplotlib.pyplot as plt
# plt.figure(figsize=(8,6))
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, vmin=-1, vmax=1)
# plt.title('Correlation Heatmap')
# plt.show()
#---------------------------------------------------------

from sklearn.metrics import r2_score
model.fit(x,y)
pred_y=model.predict(x)
# Print the R² score, which shows the proportion of the variance explained by the model
print(f'r2_score: {round(r2_score(y, pred_y),4)}') #打印出模型可以解釋的比率

#這裡，我將預測位置為 得分後衛(SG)，28歲所屬隊伍為休斯敦火箭隊(HOU)，球員的3分球進球數
# Here, I am predicting for a player with the position SG (Shooting Guard), age 28, and team Houston Rockets (HOU)
to_be_predicted=['SG',28,'HOU'] 
to_be_predicted[0]=LabelEncoder_pos.transform([to_be_predicted[0]])[0]
to_be_predicted[2]=LabelEncoder_Tm.transform([to_be_predicted[2]])[0]
to_be_predicted=np.array(to_be_predicted).reshape(1,-1)
to_be_predicted=pd.DataFrame(to_be_predicted,columns=x.columns)
#here comes result:
print(f'result: {round(model.predict(to_be_predicted)[0],4)}')
