import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import neighbors
from sklearn import tree
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
data = pd.read_csv('advertising.csv')
target='Clicked on Ad'

#將文字特徵轉換成標籤
# Convert categorical features to numeric labels
encoder_city=LabelEncoder()
data['City']=encoder_city.fit_transform(data['City'])
encoder_country=LabelEncoder()
data['Country']=encoder_country.fit_transform(data['Country'])
encoder_timestamp=LabelEncoder()
data['Timestamp']=encoder_timestamp.fit_transform(data['Timestamp'])
data=data.drop(columns=['Ad Topic Line'])
#篩選出與目標相關性較高的特徵
# Filter out features that are highly correlated with the target
corr_matrix = data.corr()
filtered_features = corr_matrix.iloc[:,:-1].columns[((corr_matrix.iloc[:,:-1] > 0.2) & (corr_matrix.iloc[:,:-1] != 1.0)).any(axis=0)]
features=corr_matrix[target]
features=features[(features.abs() < 1) & (features.abs() > 0.1) ].index

#訓練模型，用迴圈檢視不同模型的成果
# Train models and evaluate the performance of different models
x=data[features]
y=data[target]
train_X, test_X, train_y, test_y = train_test_split(x, y,test_size=0.25)
clf_dict={'RandomForestClassifier':RandomForestClassifier(),'DecisionTreeClassifier':tree.DecisionTreeClassifier(),
          'KNeighborsClassifier':neighbors.KNeighborsClassifier(),'Support vector classifier':SVC(kernel='linear', C=1)}
for clf in clf_dict:
    clf_dict[clf].fit(train_X,train_y)
    print(f'{clf}模型解釋力: {clf_dict[clf].score(train_X,train_y):.4f}')
    pred_y=clf_dict[clf].predict(test_X)
    print(f'{clf}測試集準確度: {accuracy_score(test_y,pred_y):.4f}')

