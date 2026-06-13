import pandas as pd 
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression

#Q1
df=pd.read_csv("C:\\Users\\sharm\\Downloads\\Dataset 2.csv")
print(df.head())
 
#Q2
print("Shape : ", df.shape)

#Q3
print("Columns")
print(df.columns)

#Q4
print(df.dtypes)
print(df.select_dtypes(include=['int64', 'float64']).columns)

print(df.select_dtypes(include=['object', 'category']).columns)
print(df.info())

#Q5
print(df.isnull().sum())

#Q6
print(round(df["Age"].mean(),2))

#Q7
print(round(df["WatchHoursPerWeek"].mean(),2))

#Q8
print(round(df["MonthlySpend"].mean(),2))

#Q9
print(df["SubscriptionType"].value_counts())

#Q10
renew_per=(df["SubscriptionRenewed"]=="Yes").mean()*100
print(renew_per)

#Q11
le=LabelEncoder()
df['Gender']=le.fit_transform(df['Gender'])
df['SubscriptionType']=le.fit_transform(df['SubscriptionType'])
df['FavoriteGenre']=le.fit_transform(df['FavoriteGenre'])
df['SubscriptionRenewed']=le.fit_transform(df['SubscriptionRenewed'])
print(df["Gender"].head(5))
print(df["SubscriptionType"].head(5))
print(df["FavoriteGenre"].head(5))
print(df["SubscriptionRenewed"].head(5))

#Q12
X=df.drop(['UserID','SubscriptionRenewed'],axis=1)
y=df['SubscriptionRenewed']

#Q13
X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=42
)

#Q14
dt=DecisionTreeClassifier(random_state=42)
dt.fit(X_train,y_train)
pred=dt.predict(X_test)
print(pred)

#Q15
accuracy=accuracy_score(y_test,pred)
print(f"Decision tress accuracy : {accuracy} ")

#Q16
print("CONFUSION MATRIX")
print(confusion_matrix(y_test,pred))

#Q17
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
knn_pred=knn.predict(X_test)
print(knn_pred)

#Q18
acc=accuracy_score(y_test,knn_pred)
if acc>accuracy:
    print("KNN accuracy is higher then Decision tree")
elif acc<accuracy:
    print("Accuracy of Decision tree is higher then KNN")
else:
    print("Both have accuracy")

#Q19
X_reg=df.drop(['UserID','SubscriptionRenewed','MonthlySpend'],axis=1)
y_reg=df['MonthlySpend']

X_train_reg,X_test_reg,y_train_reg,y_test_reg=train_test_split(
    X_reg,y_reg,test_size=0.2,random_state=42
)

lr=LinearRegression()
lr.fit(X_train_reg,y_train_reg)
predlr=lr.predict(X_test_reg)
print(predlr)

#Q20
new_customer = [[30, 1, 0, 20, 2, 3, 15]]
print("MonthlySpending : ",lr.predict(new_customer)[0])

#Business Reflection Questions  
# 1. Which factors appear to influence subscription renewal the most?  
#    Subscription type and Monthly spent are the factors influence the subscription renewal the most.How much a user spent
#    on subscription and which subscription type is paid by the user.
# 2. Why is subscription renewal a classification problem? 
#    Subscription is a classification problem beacuse the answer or the prediction of subscription renewed is either yes or no
#    The ouput is logical not numerical
# 3. Why is monthly spending a regression problem?  
#    Monthly spending is a regression problem because the prediction will be a complete numerical value.
#    The output will be some number not a logical value like yes or no.
# 4. Which algorithm performed better for renewal prediction? 
#    By comparing the acccuracy of Decision Tree and KNN algorithm it can be calculated which is better.
#    In this case KNN algorithm is better.
# 5. How could the platform use these predictions to improve customer retention?  
#    Netflix can identify users who are likely not to renew their subscription and
#    target them with personalised recommnedation,discounts,special offers or engagement campaigns to improve retention.