# ---------------------------------------Import Libraries-------------------
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# --------------------------------------------------------------------------
print(20*'-' + 'Q1'+20*'-')
# Q1 :
# Read the Marriage dataset and check the size and type of each features.
df = pd.read_csv('Marriage_dataset.csv')
print(df.size)
dataTypeSeries2 = df.dtypes
print(dataTypeSeries2)
df.head(6)

#----------------------------------------------------------
print(20*'-' + 'Q2'+20*'-')
# Q2 :
# Count the numbers mens and women's in the dataset
# Find the mean  and standard devation of age.
# Whtat is the minimum age and maximum age.
#index = df.index
#dic = df.groupby('sex')
#dic.agg(count = ('index', 'count'))
male = df.groupby(['sex']).get_group('male')
print('count of men are 286\n', male)
female = df.groupby(['sex']).get_group('female')
print('count of women are 315\n', female)
print(df.agg(np.min))
print(df.agg(np.max))
print(df.agg(np.mean))
print(df.agg(np.std))
print('min age: 17.5')
print('max age: 57 ')
print('mean age: 32.48')
print('std age: 9.28')


#------------------------------------------------
print(20*'-' + 'Q3'+20*'-')
# Q3 :
# Count the number of people who are below age of 35
#print(chipotle[(chipotle.quantity == 1) & (chipotle.item_price > 10)])

dic3 = df[(df.age < 35 )]
print('count of people under 35 are 391\n',dic3)


# --------------------------------------------------------------------------
print(20*'-' + 'Q4'+20*'-')
# Q4 :
# Explore the following and show a linear regression analysis for each of the following categories:
# 1- if the years of marriage is correlated to the number of affairs.
# 2- if the age is correlated to the number of affairs.
# 3- if the religion is correlated to the number of affairs.
# explain each graph in comments.


sns.regplot(x="ym", y="nbaffairs", data=df)
plt.show()
sns.regplot(x="age", y="nbaffairs", data=df)
plt.show()
sns.regplot(x="religious", y="nbaffairs", data=df)
plt.show()

sns.heatmap(df[['age','age','religious','nbaffairs']].corr(), annot = True )

print('Nba vs Ym, You can the longer the years of marriage, the more affairs are.')
print('Nba vs Age, the older the person is the most likely the person is to commit an affair')
print('Nba vs Religious, the less religious the person is the most likely he or she is to commit an affair ')
#
df.head(5)
# sns.heatmap(df.corr())
# plt.show()
# print('heatmap is the best way to see correlation between each column')
# # sns.displot(df['fare'] , color = 'red' , bins = 35)
# # plt.show()
# sns.boxplot(x='ym', y = 'nbaffairs', data = df)
# plt.show()
# sns.jointplot(x='age', y='nbaffairs', data = df)
# plt.show()
# sns.boxplot(x='religious', y = 'nbaffairs', data = df)
# plt.show()
# print('Best graph is the heatmap')


# --------------------------------------------------------------------------
print(20*'-' + 'Q5'+20*'-')
# Q5 :
# Explore the following and show a linear regression analysis for each of the following categories:
# 1- if the years of marriage and having a child is correlated to the number of affairs.
# 2- if the age and having a child is correlated  is correlated to the number of affairs.
# 3- if the religion and having a child is correlated  is correlated to the number of affairs.
# explain each graph in comments.

#sns.heatmap(df.corr())
#plt.show()
#print(' In the heatmap it shows which columns are relationships with eachother')
df['child']=df.child.replace(to_replace=['no', 'yes'], value=[0, 1])
df['child'] = df['child'].astype('int') ##reassing my copy of df to chipotle the original. i will use df later on
# df.head(6)
# sns.regplot(x="ym", y="child", data=df)
# plt.show()

sns.lmplot(x="nbaffairs", y="age",  col="child", hue="child",data=df,
           col_wrap=2, ci=None, palette="muted",
           scatter_kws={"s": 50, "alpha": 1})
plt.show()
sns.lmplot(x="nbaffairs", y="ym",  col="child", hue="child",data=df,
           col_wrap=2, ci=None, palette="muted",
           scatter_kws={"s": 50, "alpha": 1})

plt.show()
sns.lmplot(x="nbaffairs", y="religious",  col="child", hue="child",data=df,
           col_wrap=2, ci=None, palette="muted",
           scatter_kws={"s": 50, "alpha": 1})

print('Age vs Affairs has no correlation')
print('Ym vs Affairs has a positive correlation')
print('Religious vs Affairs has a negative correlation')
plt.show()



# sns.regplot(x="age", y="child", data=df)
# plt.show()
# sns.regplot(x="religious", y="child", data=df)
# plt.show()


# --------------------------------------------------------------------------
print(20*'-' + 'Q6'+20*'-')
# Q6 :
# Which features has highest correlation with other features.
# explain each graph in comments.

sns.heatmap(df.corr())
plt.show()
print('Ym and age has the highest correlation')




# --------------------------------------------------------------------------
print(20*'-' + 'Q7'+20*'-')
# Q7 :
# Lets look at education, religion and number of affairs. What can be explored from this data?
# Analysis the education level at every stage and make a judgment about the religion and education and number of affairs.
# Does it matter if you are male or female.

print('first let examine the min, max,mean of education')
print('Education min',df['education'].agg(np.min))
print('Education max',df['education'].agg(np.max))
print('Education average', df['education'].agg(np.mean))

sns.lmplot(x="education", y="religious",  col="sex", hue="sex",data=df,
           col_wrap=2, ci=None, palette="muted",
           scatter_kws={"s": 50, "alpha": 1})
print('Another interest insight in this graph, in the case of female the less educated the person is the less religious the person is, \n',
      ' there doesnt seems much of a diference in the case of male.')
plt.show()

sns.lmplot(x="education", y="nbaffairs",  col="sex", hue="sex",data=df,
           col_wrap=2, ci=None, palette="muted",
           scatter_kws={"s": 50, "alpha": 1})
print('In Nba affairs vs education you can see a diference between gender, Male has a positive line and Female has a negative line\n',
      'we are furthering analyzing this.')
plt.show()

# sns.lmplot(x="education", y="education",  col="sex", hue="sex",data=df,
#            col_wrap=2, ci=None, palette="muted",
#            scatter_kws={"s": 50, "alpha": 1})
# print('Here you can see there is no clear difference between gender for education')
# plt.show()

df.head(6)

trial = df[['education','religious','nbaffairs']]
trial.describe()
# print(dataTypeSeries2)

# df.groupby(['education']).agg(affair_average=('nbaffairs','mean')).sort_values('affair_average', ascending=False)
# df.groupby(['education','sex']).agg(affair_average=('nbaffairs', 'mean'))
print(df.groupby(['education']).agg(affair_count=('nbaffairs','count')).sort_values('affair_count', ascending=False))
print(df.groupby(['education','sex']).agg(affair_count=('nbaffairs', 'count')))
print(df.groupby(['education']).agg(religious_average=('religious','mean')))
print(df.groupby(['education','sex']).agg(religious_average=('religious', 'mean')))


# IGNORE!
# dic4.agg(count = (''))
# index = df.index
# number_of_rows = len(index)
# find length of index
# print(number_of_rows)
# dic2 = users.groupby(['occupation'])
# #type(dic2)
# min_age = dic2['age'].agg(np.min)
# max_age = dic2['age'].agg(np.max)
# data1 = pd.DataFrame(min_age).reset_index()
# data1.columns = ['occupation','min_age']
# data2 = pd.DataFrame(max_age).reset_index()
# data2.columns = ['occupation','max_age']
# maxy =list(data2.iloc[:,1]) ## maximun list
# data1['max_age'] =  maxy
# print(data1)
#
# users = pd.read_table('data.txt', sep= '|')
#
# chipotle = pd.read_table('Data2HW.txt')
# male_ratio = users.groupby(['gender']).get_group('M')
# m = male_ratio.groupby('occupation')
# final = m.agg(count = ('occupation', 'count'))
# print(final.sort_values( 'count', ascending=False))
#
#
# dic = users.groupby('occupation')
# print(dic['age'].agg(np.mean))
#
# male_ratio = users.groupby(['gender']).get_group('M')
# m = male_ratio.groupby('occupation')
# final = m.agg(count = ('occupation', 'count'))
# print(final.sort_values( 'count', ascending=False))
#
# dic2 = users.groupby(['occupation','gender'])
# before_clean = dic2.agg(age_mean=('age',np.mean))
# print(before_clean)
#
# dic2 = users.groupby(['occupation','gender'])
# trial = dic2.agg(percent=('gender','count')) ##count amount of people (occupation per gender)
# print(trial/trial.groupby(level=0).sum())
# df['item_price'] = df['item_price'].str.replace('$','')
# chipotle['item_price'] = df['item_price'].astype('float') ##reassing my copy of df to chipotle the original. i will use df later on
# print(chipotle.head(6))


# --------------------------------------------------------------------------
print(20*'-' + 'Q8'+20*'-')
# Q8 :
# Lets look at occupation, religion and number of affairs. What can be explored from this data?
# Analysis the occupation level at every stage and make a judgment about the religion and occupation and number of affairs.
# Does it matter if you are male or female.

print('First let examine the min, max,mean of occupation')
print('Occupation min',df['occupation'].agg(np.min))
print('Occupation max',df['occupation'].agg(np.max))
print('Occupation average', df['occupation'].agg(np.mean))

sns.lmplot(x="occupation", y="religious",  col="sex", hue="sex",data=df,
           col_wrap=2, ci=None, palette="muted",
           scatter_kws={"s": 50, "alpha": 1})
print('In the case of Female we can see in this graph the higher the occupation the less religious the person is , \n',
      ' there doesnt seems much of a diference in the case of male. Religious vs Occupation the gender matters')
plt.show()

sns.lmplot(x="occupation", y="nbaffairs",  col="sex", hue="sex",data=df,
           col_wrap=2, ci=None, palette="muted",
           scatter_kws={"s": 50, "alpha": 1})
print('In Nba affairs vs occupation you can see a diference between gender, Male has a positive line and in Female\n',
      'nba vs occupation has no relationship. Nbaaffairs vs ocuppation the gender matters. ')
plt.show()

print(df.groupby(['occupation']).agg(affair_count=('nbaffairs','count')).sort_values('affair_count', ascending=False))
print(df.groupby(['occupation','sex']).agg(affair_count=('nbaffairs', 'count')))
print(df.groupby(['education']).agg(religious_average=('religious','mean')))
print(df.groupby(['education','sex']).agg(religious_average=('religious', 'mean')))

df.tail(6)
# sns.lmplot(x="occupation", y="occupation",  col="sex", hue="sex",data=df,
#            col_wrap=2, ci=None, palette="muted",
#            scatter_kws={"s": 50, "alpha": 1})
# print('Here you can see there is no clear difference between gender for education')
# plt.show()




