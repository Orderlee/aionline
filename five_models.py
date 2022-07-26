from sklearn.datasets import make_regression
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn import metrics
from sklearn.metrics import f1_score

df = pd.read_csv('/Users/lyw/baseline/train/train.csv')
df['leaktype'] = df['leaktype'].replace(['in'],0)
df['leaktype'] = df['leaktype'].replace(['noise'],1)
df['leaktype'] = df['leaktype'].replace(['normal'],2)
df['leaktype'] = df['leaktype'].replace(['other'],3)
df['leaktype'] = df['leaktype'].replace(['out'],4)

dataset = df.values
X = dataset[:,1:]
print(X.shape)
Y = dataset[:,0]
print(Y)
min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)
# train, val = train_test_split(df, test_size=0.2, random_state=100)
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=100)
X_train, X_test, Y_train, Y_test = train_test_split(X_scale, Y, test_size=0.2, random_state=100)

print('-----------------------------------------------')

# fit final model
# Gaussian NB
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
Y_train = Y_train.astype('int')
model.fit(X_train, Y_train)
ynew = model.predict(X_test)
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = model, X = X_train, y = Y_train, cv = 10)
print("Gaussian NB Accuracy(cross validation):",accuracies.mean())
print("Gaussian NB Accuracy Score:", model.score(X_test,Y_test))
print('F1 Score:',f1_score(Y_test, ynew, average=None))
print('Macro F1 Score:',f1_score(Y_test, ynew, average='macro'),'\n')


from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt

# label=['in', 'noise', 'normal', 'other', 'out'] # 라벨 설정
# plot = plot_confusion_matrix(model, # 분류 모델
#                              ynew, Y_test, # 예측 데이터와 예측값의 정답(y_true)
#                              display_labels=label, # 표에 표시할 labels
#                              cmap=plt.cm.Blue, # 컬러맵(plt.cm.Reds, plt.cm.rainbow 등이 있음)
#                              normalize=None) # 'true', 'pred', 'all' 중에서 지정 가능. default=None
# plot.ax_.set_title('Confusion Matrix')

from sklearn.metrics import confusion_matrix,accuracy_score, recall_score, precision_score, f1_score, matthews_corrcoef

pred=model.predict(X_test)
cm=confusion_matrix(Y_test,pred)
df_cm=pd.DataFrame(cm,index=['in', 'noise','normal','other','out'],columns=['in', 'noise','normal','other','out'])
print(df_cm)

print('-----------------------------------------------')


# LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
model = LinearDiscriminantAnalysis()
Y_train = Y_train.astype('int')
model.fit(X_train, Y_train)
ynew = model.predict(X_test)
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = model, X = X_train, y = Y_train, cv = 10)
print("LDA Accuracy(cross validation):",accuracies.mean())
print("LDA Accuracy Score:", model.score(X_test,Y_test))
print('F1 Score:',f1_score(Y_test, ynew, average=None))
print('Macro F1 Score:',f1_score(Y_test, ynew, average='macro'),'\n')

print('-----------------------------------------------')


# Logistic Regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
Y_train = Y_train.astype('int')
model.fit(X_train, Y_train)
ynew = model.predict(X_test)
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = model, X = X_train, y = Y_train, cv = 10)
print("Logistic Regression Accuracy(cross validation):",accuracies.mean())
print("Logistic Regression Accuracy Score:", model.score(X_test,Y_test))
print('F1 Score:',f1_score(Y_test, ynew, average=None))
print('Macro F1 Score:',f1_score(Y_test, ynew, average='macro'),'\n')

print('-----------------------------------------------')


# Decision_Tree_Classifier
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion="entropy", max_depth=100000)
Y_train = Y_train.astype('int')
model.fit(X_train, Y_train)
ynew = model.predict(X_test)
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = model, X = X_train, y = Y_train, cv = 10)
print("Decision Tree Accuracy(cross validation):",accuracies.mean())
print("Decision Tree Accuracy Score:", model.score(X_test,Y_test))
print('F1 Score:',f1_score(Y_test, ynew, average=None))
print('Macro F1 Score:',f1_score(Y_test, ynew, average='macro'),'\n')


print('-----------------------------------------------')


# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(max_depth=100, random_state=0)
Y_train = Y_train.astype('int')
model.fit(X_train, Y_train)
ynew = model.predict(X_test)
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = model, X = X_train, y = Y_train, cv = 10)
print("Random Forest Accuracy(cross validation):",accuracies.mean())
print("Random Forest Accuracy Score:", model.score(X_test,Y_test))
print('F1 Score:',f1_score(Y_test, ynew, average=None))
print('Macro F1 Score:',f1_score(Y_test, ynew, average='macro'),'\n')