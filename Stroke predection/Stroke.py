import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
Data=pd.read_csv("D:\VS codes\machine learning\ml data.csv")
print(Data)
print('\n data size',Data.shape)
print('the columns of data \n',Data.columns)
print(Data.head())
print(Data.info())
print('Number of duplicated row=',Data.duplicated().sum())
Data= Data.drop({"id","work_type"},axis=1)
print(Data.head(0))
bmi_mean=Data['bmi'].mean()
Data['bmi']=Data['bmi'].fillna(bmi_mean)
Data['gender']=Data['gender'].replace({'Other':'Male'})
print(Data.describe())
print(Data.isnull().sum())

plt.figure(figsize=(10, 5))
sns.histplot(Data["age"], bins=10, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(10, 5))
sns.countplot(x="gender", data=Data)
plt.title("Gender Count")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.show()


plt.figure(figsize=(10, 5))
sns.boxplot(x="smoking_status", y="bmi", data=Data)
plt.title("BMI Distribution by Smoking Status")
plt.xlabel("Smoking Status")
plt.ylabel("BMI")
plt.show()

plt.figure(figsize=(10, 5))
sns.countplot(x="stroke",data=Data)
mean_stroke = Data["stroke"].mean()
plt.axhline(mean_stroke, color='red', linestyle='--', label='Mean')
plt.legend()
plt.show()

plt.figure(figsize=(8, 8))
Data["smoking_status"].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title("Smoking Status Distribution")
plt.ylabel("")
plt.show()

plt.figure(figsize=(10, 5))
sns.scatterplot(x="age", y="avg_glucose_level", data=Data)
plt.title("Age vs. Average Glucose Level")
plt.xlabel("Age")
plt.ylabel("Average Glucose Level")
plt.show()
from sklearn import preprocessing
Data_types=Data.dtypes
for i in range(Data.shape[1]):
    if Data_types[i]=="O":
        pr_data=preprocessing.LabelEncoder()
        Data[Data.columns[i]]=pr_data.fit_transform(Data[Data.columns[i]])
        print(i)
        print(pr_data.classes_)

plt.figure(figsize=(10, 8))
sns.heatmap(Data.corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()



X = Data.iloc[:,:-1]
Y = Data.iloc[:,-1]
x_train, x_test,y_train,y_test= train_test_split(X,Y, test_size=0.2)
print("x_train =", x_train)
print("x_test =", x_test)

log=LogisticRegression()
l=log.fit(x_train,y_train)
print(log.coef_)
print(log.intercept_)
pr=l.predict(x_test)
print(pr)
print(l.score(x_test, y_test))
y_true = y_test
y_pre=pr
con=confusion_matrix(y_true,y_pre)
print(con)

ss=SVC(kernel="linear")
c=ss.fit(x_train, y_train)
acc=ss.score(x_test, y_test)
pre=c.predict(x_test)
print(pre)
y_true = y_test
y_pre=pr
con1=confusion_matrix(y_true,y_pre)
print(con1)

