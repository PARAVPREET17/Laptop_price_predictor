
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('laptop_data.csv')

df.head(5)

df.isnull().sum() # to check any null values in any column

df.duplicated().sum() # to check any duplicate rows so that we can remove

df.drop(columns=['Unnamed: 0'],inplace=True) #drop unwanted columns

df.head()

df['Ram']=df['Ram'].str.replace('GB','')

df['Weight']=df['Weight'].str.replace('kg','')

df['Ram']=df['Ram'].astype('int32')
df['Weight']=df['Weight'].astype('float64')

import seaborn as sns
sns.distplot(df['Price'])

df['Company'].value_counts().plot(kind='bar')

sns.barplot(x=df['Company'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()

df['TypeName'].value_counts().plot(kind='bar')

sns.barplot(x=df['TypeName'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()

sns.scatterplot(x=df['Inches'],y=df['Price'])

df['ScreenResolution'].value_counts()

df['Touchscreen']=df['ScreenResolution'].apply(lambda x:1 if 'Touchscreen' in x else 0)

df['Touchscreen'].value_counts().plot(kind='bar')

sns.barplot(x=df['Touchscreen'],y=df['Price'])

df['IPS']=df['ScreenResolution'].apply(lambda x:1 if 'IPS' in x else 0)
df['IPS'].value_counts().plot(kind='bar')

sns.barplot(x=df['IPS'],y=df['Price'])

new=df['ScreenResolution'].str.split('x',n=1,expand=True)

df['X_res']=new[0]
df['Y_res']=new[1]

df.sample()

df['X_res']=df['X_res'].str.replace(',','').str.findall(r'(\d+\.?\d)').apply(lambda x:x[0])

df['X_res']=df['X_res'].astype('int')
df['Y_res']=df['Y_res'].astype('int')

df.info()

df.corr()['Price']

df['ppi']=(((df['X_res']**2)+(df['Y_res']**2))**0.5/df['Inches']).astype('float')

df.drop(columns=['ScreenResolution','X_res','Y_res','Inches'],inplace=True)

df.head()

df['CPU Name']=df['Cpu'].apply(lambda x:' '.join(x.split()[0:3]))

df.head()

def fetch_processor(text):
  if text == 'Intel Core i5' or text == 'Intel Core i7' or text == 'Intel Core i3' :
    return text
  else:
    if text.split()[0]=='Intel':
        return 'Other Intel Processor'
    else:
        return 'AMD Processor'

df['CPU Name']=df['CPU Name'].apply(fetch_processor)

df['CPU Name'].value_counts()

df.drop(columns=['Cpu'],inplace=True)

df.head()

df['Ram'].value_counts().plot(kind='bar')

sns.barplot(x=df['Ram'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()

df['Memory'].value_counts()

df['Memory'] = df['Memory'].astype(str).replace('\.0', '', regex=True)
df["Memory"] = df["Memory"].str.replace('GB', '')
df["Memory"] = df["Memory"].str.replace('TB', '000')
new = df["Memory"].str.split("+", n = 1, expand = True)

df["first"]= new[0]
df["first"]=df["first"].str.strip()

df["second"]= new[1]

df["Layer1HDD"] = df["first"].apply(lambda x: 1 if "HDD" in x else 0)
df["Layer1SSD"] = df["first"].apply(lambda x: 1 if "SSD" in x else 0)
df["Layer1Hybrid"] = df["first"].apply(lambda x: 1 if "Hybrid" in x else 0)
df["Layer1Flash_Storage"] = df["first"].apply(lambda x: 1 if "Flash Storage" in x else 0)

df['first'] = df['first'].str.replace(r'\D', '')

df["second"].fillna("0", inplace = True)

df["Layer2HDD"] = df["second"].apply(lambda x: 1 if "HDD" in x else 0)
df["Layer2SSD"] = df["second"].apply(lambda x: 1 if "SSD" in x else 0)
df["Layer2Hybrid"] = df["second"].apply(lambda x: 1 if "Hybrid" in x else 0)
df["Layer2Flash_Storage"] = df["second"].apply(lambda x: 1 if "Flash Storage" in x else 0)

df['second'] = df['second'].str.replace(r'\D', '')

df["first"] = df["first"].astype(int)
df["second"] = df["second"].astype(int)

df["HDD"]=(df["first"]*df["Layer1HDD"]+df["second"]*df["Layer2HDD"])
df["SSD"]=(df["first"]*df["Layer1SSD"]+df["second"]*df["Layer2SSD"])
df["Hybrid"]=(df["first"]*df["Layer1Hybrid"]+df["second"]*df["Layer2Hybrid"])
df["Flash_Storage"]=(df["first"]*df["Layer1Flash_Storage"]+df["second"]*df["Layer2Flash_Storage"])

df.drop(columns=['first', 'second', 'Layer1HDD', 'Layer1SSD', 'Layer1Hybrid',
       'Layer1Flash_Storage', 'Layer2HDD', 'Layer2SSD', 'Layer2Hybrid',
       'Layer2Flash_Storage'],inplace=True)

df.drop(columns=['Memory'],inplace=True)

df.head()

df.corr()['Price']

df.drop(columns=['Flash_Storage','Hybrid'],inplace=True)

df['GPU brand']=df['Gpu'].apply(lambda x:x.split()[0])

df.head()

df.drop(columns=['Gpu'],inplace=True)

df=df[df['GPU brand'] != 'ARM']

df['OpSys'].value_counts()

sns.barplot(x=df['OpSys'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()

def cat_os(inp):
  if inp=='Windows 10' or inp=='Windows 10 S' or inp=='Windows 7':
    return 'Windows'
  elif inp=='macOS' or inp=='Mac OS X':
    return 'macOS'
  else:
   return 'Others/No OS/Linux'

df['OpSys']=df['OpSys'].apply(cat_os)

df.head()

sns.barplot(x=df['OpSys'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()

sns.heatmap(df.corr())

x=df.drop(columns=['Price'])
y=np.log(df['Price'])

from sklearn.model_selection import  train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.15,random_state=2)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score,mean_absolute_error
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor,ExtraTreesRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

# step1 = ColumnTransformer(transformers=[('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,3,8,11])],remainder='passthrough')

# step2=LinearRegression()

# pipe=Pipeline(
#     [('step1',step1),
#      ('step2',step2)]
# )
# pipe.fit(x_train,y_train)
# y_pred=pipe.predict(x_test)

# print('R2 Score',r2_score(y_test,y_pred))
# print('MAE',mean_absolute_error(y_test,y_pred))

# step1 = ColumnTransformer(transformers=[('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,3,8,11])],remainder='passthrough')

# step2=Ridge(alpha=10)

# pipe=Pipeline(
#     [('step1',step1),
#      ('step2',step2)]
# )
# pipe.fit(x_train,y_train)
# y_pred=pipe.predict(x_test)

# print('R2 Score',r2_score(y_test,y_pred))
# print('MAE',mean_absolute_error(y_test,y_pred))

# step1 = ColumnTransformer(transformers=[('col_tnf',OneHotEncoder(handle_unknown='ignore',sparse=False,drop='first'),[0,1,3,8,11])],remainder='passthrough')

# step2=Lasso(alpha=0.001)

# pipe=Pipeline(
#     [('step1',step1),
#      ('step2',step2)]
# )
# pipe.fit(x_train,y_train)
# y_pred=pipe.predict(x_test)

# print('R2 Score',r2_score(y_test,y_pred))
# print('MAE',mean_absolute_error(y_test,y_pred))

# step1 = ColumnTransformer(transformers=[('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,3,8,11])],remainder='passthrough')

# step2=KNeighborsRegressor(n_neighbors=3)

# pipe=Pipeline(
#     [('step1',step1),
#      ('step2',step2)]
# )
# pipe.fit(x_train,y_train)
# y_pred=pipe.predict(x_test)

# print('R2 Score',r2_score(y_test,y_pred))
# print('MAE',mean_absolute_error(y_test,y_pred))

# step1 = ColumnTransformer(transformers=[('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,3,8,11])],remainder='passthrough')

# step2=DecisionTreeRegressor(max_depth=8)

# pipe=Pipeline(
#     [('step1',step1),
#      ('step2',step2)]
# )
# pipe.fit(x_train,y_train)
# y_pred=pipe.predict(x_test)

# print('R2 Score',r2_score(y_test,y_pred))
# print('MAE',mean_absolute_error(y_test,y_pred))

step1 = ColumnTransformer(transformers=[('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,3,8,11])],remainder='passthrough')

step2=RandomForestRegressor(n_estimators=100,random_state=3,max_samples=0.5,max_features=0.75,max_depth=15)

pipe=Pipeline(
    [('step1',step1),
     ('step2',step2)]
)
pipe.fit(x_train,y_train)
y_pred=pipe.predict(x_test)

print('R2 Score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))

# step1 = ColumnTransformer(transformers=[('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,3,8,11])],remainder='passthrough')

# step2=SVR(kernel='rbf',C=10000,epsilon=0.1)

# pipe=Pipeline(
#     [('step1',step1),
#      ('step2',step2)]
# )
# pipe.fit(x_train,y_train)
# y_pred=pipe.predict(x_test)

# print('R2 Score',r2_score(y_test,y_pred))
# print('MAE',mean_absolute_error(y_test,y_pred))

# step1 = ColumnTransformer(transformers=[('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,3,8,11])],remainder='passthrough')

# step2=ExtraTreesRegressor(n_estimators=100,random_state=3,max_features=0.75,max_depth=15)

# pipe=Pipeline(
#     [('step1',step1),
#      ('step2',step2)]
# )
# pipe.fit(x_train,y_train)
# y_pred=pipe.predict(x_test)

# print('R2 Score',r2_score(y_test,y_pred))
# print('MAE',mean_absolute_error(y_test,y_pred))

# step1 = ColumnTransformer(transformers=[('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,3,8,11])],remainder='passthrough')

# step2=AdaBoostRegressor(n_estimators=15,learning_rate=1.0)

# pipe=Pipeline(
#     [('step1',step1),
#      ('step2',step2)]
# )
# pipe.fit(x_train,y_train)
# y_pred=pipe.predict(x_test)

# print('R2 Score',r2_score(y_test,y_pred))
# print('MAE',mean_absolute_error(y_test,y_pred))

# step1 = ColumnTransformer(transformers=[('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,3,8,11])],remainder='passthrough')

# step2=GradientBoostingRegressor(n_estimators=500)

# pipe=Pipeline(
#     [('step1',step1),
#      ('step2',step2)]
# )
# pipe.fit(x_train,y_train)
# y_pred=pipe.predict(x_test)

# print('R2 Score',r2_score(y_test,y_pred))
# print('MAE',mean_absolute_error(y_test,y_pred))



# step1 = ColumnTransformer(transformers=[('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,3,8,11])],remainder='passthrough')

# step2=XGBRegressor(n_estimators=45,learning_rate=0.5,max_depth=5)

# pipe=Pipeline(
#     [('step1',step1),
#      ('step2',step2)]
# )
# pipe.fit(x_train,y_train)
# y_pred=pipe.predict(x_test)

# print('R2 Score',r2_score(y_test,y_pred))
# print('MAE',mean_absolute_error(y_test,y_pred))

# from sklearn.ensemble import VotingRegressor,StackingRegressor

# step1 = ColumnTransformer(transformers=[
#     ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,3,8,11])
# ],remainder='passthrough')


# rf = RandomForestRegressor(n_estimators=350,random_state=3,max_samples=0.5,max_features=0.75,max_depth=15)
# gbdt = GradientBoostingRegressor(n_estimators=100,max_features=0.5)
# xgb = XGBRegressor(n_estimators=25,learning_rate=0.3,max_depth=5)
# et = ExtraTreesRegressor(n_estimators=100,random_state=3,max_features=0.75,max_depth=10)

# step2 = VotingRegressor([('rf', rf), ('gbdt', gbdt), ('xgb',xgb), ('et',et)],weights=[5,1,1,1])

# pipe = Pipeline([
#     ('step1',step1),
#     ('step2',step2)
# ])

# pipe.fit(x_train,y_train)

# y_pred = pipe.predict(x_test)

# print('R2 score',r2_score(y_test,y_pred))
# print('MAE',mean_absolute_error(y_test,y_pred))

# from sklearn.ensemble import VotingRegressor,StackingRegressor

# step1 = ColumnTransformer(transformers=[
#     ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,3,8,11])
# ],remainder='passthrough')


# estimators = [
#     ('rf', RandomForestRegressor(n_estimators=350,random_state=3,max_samples=0.5,max_features=0.75,max_depth=15)),
#     ('gbdt',GradientBoostingRegressor(n_estimators=100,max_features=0.5)),
#     ('xgb', XGBRegressor(n_estimators=25,learning_rate=0.3,max_depth=5))
# ]

# step2 = StackingRegressor(estimators=estimators, final_estimator=Ridge(alpha=100))

# pipe = Pipeline([
#     ('step1',step1),
#     ('step2',step2)
# ])

# pipe.fit(x_train,y_train)

# y_pred = pipe.predict(x_test)

# print('R2 score',r2_score(y_test,y_pred))
# print('MAE',mean_absolute_error(y_test,y_pred))

import pickle

pickle.dump(df,open('df.pkl','wb'))
pickle.dump(pipe,open('pipe.pkl','wb'))