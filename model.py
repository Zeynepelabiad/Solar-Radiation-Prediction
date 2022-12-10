import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn import metrics

columns=['Unixtime','date','hour','radiation','temperature','pressure','humidity',
'winddirection','speed','sunrise','sunset']
df = pd.read_csv('SolarPrediction.csv',header=0,names=columns)
df = df.loc[(df['hour'] >= '06:00:00') & (df['hour'] <= '17:00:00')]


df.drop(['Unixtime'], axis=1, inplace=True)
df['date']=df[['date', 'hour']]. T. agg(''. join)
df['date']= pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
df=df.sort_index().truncate(before='2016-09-01 00:00:00', after='2016-12-31 23:00:00')
df = df.loc[(df['hour'] >= '05:00:00') & (df['hour'] <= '17:00:00')]


plt.rcParams["figure.figsize"] = [8, 6]
ax = df[['hour','radiation','temperature','pressure','humidity',
'winddirection','speed','sunrise','sunset']].plot(kind='box', title='boxplot comparison - outliers')
plt.ylabel("value")
plt.savefig('static/assets/img/boxplot.png')


fig, ax = plt.subplots()
df['radiation'].plot(ax=ax, color='orange')
ax.set_title('Radiation')
ax.set_ylabel('Radiation')
plt.savefig('static/assets/img/timeplot1.png')

fig, ax = plt.subplots()
df['temperature'].plot(ax=ax, color='green')
ax.set_title('Temperature')
ax.set_ylabel('Temperature')
plt.savefig('static/assets/img/timeplot2.png')



plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(),annot=True,cmap='viridis',linewidths=.1)
plt.savefig('static/assets/img/heatmap.png')



Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]

X= df.drop(['hour','radiation','sunrise','sunset'],axis=1).values
y=df['radiation'].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)


from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()

#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)


model=RandomForestRegressor(min_samples_leaf=2)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.plot(range(1500), range(1500))
plt.savefig('static/assets/img/prediction.png')


print('R^2: %.3f' % r2_score(y_test, y_pred))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

pickle.dump(model, open('model.pkl','wb'))

