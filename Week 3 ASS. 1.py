import pandas as pd 
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.svm import SVC

#Q1
df=pd.read_csv("C:\\Users\\sharm\\Downloads\\agriculture_yield_dataset (1).csv")
print(df.shape) # rows are 1500 and columns are 8
print(df.columns) # column names with dtype - ['rainfall_mm', 'temperature_c', 'fertilizer_kg', 'irrigation_hours', 'soil_ph', 'crop_type', 'soil_type', 'yield_ton_per_hectare']
print(df.head(10)) # dataframe of first 10 rows

#Q2
print(df.dtypes)
print(df.isnull().values.any())
print(df.isnull().sum())

#Q3
std_values=df.std(numeric_only=True)
print(std_values)
max_std=std_values.idxmax()
print(f"The highest standard deviation is {max_std}")
mean_values=df.mean(numeric_only=True)
print(mean_values)
max_mean=mean_values.idxmax()
print(f'The highest mean value is {max_mean}')

#Q4
fig=px.histogram(df,x='rainfall_mm',marginal='box',color_discrete_sequence=['red'],title='HISTOGRAM OF RAINFALL')
fig.update_layout(bargap=0.2)
fig.show()
# From histogram it can be concluded that the hightest value of rainfall in mm is from between 825-874.9 about 101
# Almost similar value of count from 425 to 724.9
fig=px.histogram(df,x='temperature_c',marginal='box',color_discrete_sequence=['blue'],title='HISTOGRAM OF TEMPERATURE')
fig.update_layout(bargap=0.2)
fig.show()
# Maximum crop growth is between the temperature 21.5 to 22.4 degree celcius.
# Minimum crop growth is between the temperature range of 37.5 - 38.4.
# There is no outlier present in the data of temperature and approx average temperatur eis 27 degree celcius.
fig=px.histogram(df,x='fertilizer_kg',marginal='box',color_discrete_sequence=['green'],title='HISTOGRAM OF fertilizer_kg ')
fig.update_layout(bargap=0.2)
fig.show()
# NO outliers in the data of fertilizer_kg.
# Maximum fertilizer used in killo grams is 75-84.9.
# No uniform distribution pattern

fig=px.histogram(df,x='yield_ton_per_hectare',marginal='box',color_discrete_sequence=['pink'],title='HISTOGRAM OF yield_ton_per_hectare ')
fig.update_layout(bargap=0.2)
fig.show()
# This graph reflects GAUSSAIN DISTRIBUTION.
# The highest frequency is between the range of 4.8 to 5.
# The highest count is 119 and 3 outliers one maximum side and other two on minimum side.

#Q5 
crop_count=df['crop_type'].value_counts()
print(crop_count)

crop_counts=df['crop_type'].value_counts().reset_index()
crop_counts.columns = ["crop_type", "crop_count"]
fig=px.bar(crop_counts,x='crop_type',y='crop_count',color_discrete_sequence=['red'],title='Bar Chart of Count of crops')
fig.update_layout(xaxis_title='CROP TYPE' ,yaxis_title='COUNT OF CROPS')
fig.show()
# The crop occured most frequently is COTTON which 311 in count.

#Q6
soil_count=df['soil_type'].value_counts()
print(soil_count)

soil_counts=df['soil_type'].value_counts().reset_index()
soil_counts.columns=['Soil','Frequency']
fig=px.bar(soil_counts,x='Soil',y='Frequency',color_discrete_sequence=['green'],title="BAR PLOT of Soil type")
fig.update_layout(xaxis_title='Soil Type',yaxis_title='Frequency')
fig.show()
# CLAY is the soil type which is the most common type.

#Q7
fig=px.histogram(df,x='yield_ton_per_hectare',marginal='box',color_discrete_sequence=['pink'],title='HISTOGRAM OF yield_ton_per_hectare ')
fig.update_layout(bargap=0.2)
fig.show()
# YES from the histogram of yield_ton_per_hectare it can be concluded that it is a approximately normal distribution
# YES there are some noticebale outliers which are 3 - 1 at the extreme of maximum and 2 at the extreme of minimum.

#Q8
plt.title('rainfall_mm vs yield_ton_per_hectare')
sns.scatterplot(data=df, x='rainfall_mm', y='yield_ton_per_hectare', alpha=0.7, s=15);
plt.show()

plt.title('fertilizer_kg vs yield_ton_per_hectare')
sns.scatterplot(data=df, x='fertilizer_kg', y='yield_ton_per_hectare', alpha=0.7, s=15);
plt.show()

# On comparing the relationship of rainfall and fertilizer with yield it can be concluded that rainfall have strong relationship with yield.

#Q9
le = LabelEncoder()
df['crop_type'] = le.fit_transform(df['crop_type']) 
df['soil_type'] = le.fit_transform(df['soil_type']) 
sns.heatmap(df.corr(), cmap='Reds', annot=True)
plt.title('Correlation Matrix');
plt.show()

# Three features most correlated with Crop Yield are :-
# 1) rainfall_mm (0.55)
# 2) irrigation_hours (0.54)
# 3) fertilizer_kg (0.28)

#Q10
mean_soil=df.groupby('soil_type')['yield_ton_per_hectare'].mean()
print(mean_soil)

mean_crop=df.groupby('crop_type')['yield_ton_per_hectare'].mean()
print(mean_crop)

# Crop type have highest average yield - RICE
# Soil type have highest average yield - LOAMY

#Q11
print(df.dtypes) # from the output it can be analysed that crop_type and soil_type is categorical data."""

converted_data=pd.get_dummies(df,columns=['soil_type','crop_type'],dtype=int)
print(converted_data.head())

#Q12
X=converted_data.drop('yield_ton_per_hectare',axis=1)
Y=converted_data['yield_ton_per_hectare']
# Here Y - is the target variable , using yield_ton_per_hectare column as target for predicting
# X is the INPUT features

#Q13
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
print("X_train shape : ",X_train.shape)
print("X_test shape : ",X_test.shape)
print("Y_train shape : ",Y_train.shape)
print("X_test shape : ",X_test.shape)

#Q14
lr_model=LinearRegression()
lr_model.fit(X_train,Y_train)

print("Coefficients : ",lr_model.coef_)
print("Intercept : ",lr_model.intercept_)

# The highest positive coefficient is of crop type RICE