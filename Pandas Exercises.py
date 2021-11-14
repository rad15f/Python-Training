##RICARDO DIAZ
##HW6

# E.1:
# Work with Pandas module and answer the following questions. Open a .py file and follow the
# instructions and write codes for each section.
# i. Import Pandas and libraries that you think it is needed.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# ii. Import the dataset from BB. The name of the dataset is Data2.txt.
chipotle = pd.read_table('Data2HW.txt')

#print(chipotle)

# iii. Assign it to a variable called chipotle and print the 6 observation of it.
print(chipotle.head(6))
# iv. Clean the item price column and change it in a float data type then reassign the column with
# the cleaned prices.
df = chipotle.copy()

df['item_price'] = df['item_price'].str.replace('$','')
chipotle['item_price'] = df['item_price'].astype('float') ##reassing my copy of df to chipotle the original. i will use df later on
print(chipotle.head(6))
#dataTypeSeries = chipotle.dtypes
##print(dataTypeSeries)

#type(chipotle['item_price'])

# v. Remove the duplicates in item name and quantity.

chipotle.drop_duplicates(subset=['item_name','quantity'], keep='first', inplace = True) ##removing duplicates



##df.drop_duplicates(subset='quantity', keep='first', inplace = True)
##chiptole.drop(chiptole[chipotle['quantity'] > 2].index, inplace = True)
# pd.set_option('display.max_rows', 10)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)

# vi. Find only the products with quantity equals to 1 and find item price that greater that 10.

print(chipotle[(chipotle.quantity == 1) & (chipotle.item_price > 10)])

##  data[(data['phylum'].str.endswith('bacteria')) & (data['value'] > 1000)]

# vii. Find the price of each item.

df.drop_duplicates(subset='item_name', keep='first', inplace = True) #droping any duplicates of item name and keeping the first, using my df copy from chipotle
print(df[['item_name','item_price']])
# dataTypeSeries = df.dtypes
# print(dataTypeSeries)


# viii. Sort by the name of the item.

t = chipotle.sort_values(by='item_name') ##sorting, i assigned to an variable for my clarity
print(t)

# ix. find the most expensive item ordered.


#t.max()
t.item_price.max() ##checking the highest price
t.item_price.idxmax() ## checking the index

print('item name: ', t.item_name[3598],' price: ',t.item_price[3598])

# x. Find the frequency of times were a Veggie Salad Bowl ordered.

# dups = chipotle.pivot_table(index = ['item_name'], aggfunc ='size')
# d

counts = chipotle['item_name'].value_counts().to_dict() #counting every item name from chipotle with the duplicates dropped
print('The frequency of times were Veggie Salad Bowl where', counts['Veggie Salad Bowl']) #only taking veggie salad


# xi. How many times people ordered more than one Canned Soda?

canned_soda = chipotle[(chipotle['item_name'].str.endswith('Soda')) & (chipotle['quantity'] > 1)] ## assigning the datafram with conditions to a variable so i can print the rows

print('Frequency of Canned Soda > 1: ',canned_soda.shape[0]) #priting the row of my canned soda dataframe


# data[(data['phylum'].str.endswith('bacteria')) & (data['value'] > 1000)]
# canned_soda = chipotle['item_value']
# canned_soda = chipotle.groupby('order_id')
# canned_soda.first()
#chipotle.head()
# chipotle = pd.read_table('Data2HW.txt')
# print(chipotle)
# chipotle.head(6)
# df = chipotle.copy()
# df['item_price']

# Work with Pandas module and answer the following questions. Open a .py file and follow the
# instructions and write codes for each section.
# i. Import Pandas and libraries that you think it is needed.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# ii.Import the dataset from BB. The name of the dataset is Food.txt.

food = pd.read_table('Food.tsv')
#food.head(4)

# foody = pd.read_csv("Food.tsv", sep='\t')
# print(foody)

# iii. Print the size of the data frame and then 6 observation of it.

print('size of the dataframe',food.size)
print(food.head(6))


# iv. How many columns this dataset has and print the name of all the columns.

print('Numbers of Columns',food.shape[1])

print(food.columns)
# v. What is the name and data type of 105th column?

print('Name of Column:' ,food.columns[105])
print('Type of Column: ',food.dtypes[105])

# dataTypeSeries2 = food.dtypes
# print(dataTypeSeries2)

# vi. What are the indices of this datset. How they are shaped and ordered.

index = food.index
print(index)
print('Number of columns: ',food.shape[1])
print('Number of rows', food.shape[0])
print('Indices are order first with rows and the columns, they are shape rows X columns')
# vii. What is the name of product of 100th observation .
print(food.columns[100])

print('product name:',food.loc[100,'product_name'])

# E.3:
# Work with Pandas module and answer the following questions. Open a .py file and follow the
# instructions and write codes for each section.
# i. Import Pandas and libraries that you think it is needed.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ii. Import the dataset from BB. The name of the dataset is Data.txt.

users = pd.read_table('data.txt', sep= '|')

#users.index.name = 'users' this put a name to the index list

# iii. Assign it to a variable called users and print the 6 observation of it.

print(users.head(6))

# iv. Find what is the mean age for occupation.

# canned_soda = chipotle.groupby('order_id')

dic = users.groupby('occupation')
print(dic['age'].agg(np.mean))


# v. Find the male ratio for occupation and sort it from the most to the least.

##WELPPPP
##print(df.groupby(['Team','Year']).groups)
## where / fitler = male , etc ideas
# filter =
male_ratio = users.groupby(['gender']).get_group('M')
m = male_ratio.groupby('occupation')
final = m.agg(count = ('occupation', 'count'))
print(final.sort_values( 'count', ascending=False))

#m =  male_ratio.groupby('occupation').agg(count = ('occupation', 'count'))

# dic2.agg(np.mean)
# dic2['age'].agg(np.mean)
#df[1] = df[1].apply(add_one) FUNCTION TO CALCULATE RATIO ON THE MEN TABLE



# vi. For each occupation, calculate the minimum and maximum ages.
dic2 = users.groupby(['occupation'])
#type(dic2)
min_age = dic2['age'].agg(np.min)
max_age = dic2['age'].agg(np.max)

data1 = pd.DataFrame(min_age).reset_index()
data1.columns = ['occupation','min_age']
data2 = pd.DataFrame(max_age).reset_index()
data2.columns = ['occupation','max_age']
maxy =list(data2.iloc[:,1]) ## maximun list
data1['max_age'] =  maxy
print(data1)


# df = pd.DataFrame(s).reset_index()
# df.columns = ['Gene', 'count']
# df
# type(data2)
# type(data2['max_age'])


# vii.For each combination of occupation and gender, calculate the mean age.
dic2 = users.groupby(['occupation','gender'])
before_clean = dic2.agg(age_mean=('age',np.mean))
print(before_clean)
# clean = pd.DataFrame(before_clean).reset_index()
# clean.columns = ['occupation','gender','mean_age']
# clean
#dic2.agg(np.mean)
# before_clean = dic2.agg(age_mean=('age',np.mean))
# print(before_clean)

# viii. Per occupation present the percentage of women and men.


dic2 = users.groupby(['occupation','gender'])
trial = dic2.agg(percent=('gender','count')) ##count amount of people (occupation per gender)
print(trial/trial.groupby(level=0).sum()) ## sum each row and divide by their sum


# dic4 = users.groupby(['occupation'])
# dic4['gender'].value_counts(normalize = True)
##trial = dic2['age'].agg(np.percentile(100)) ## doesnt wrong
# dic2['age'].groups
# dic4 = users.groupby(['occupation'])
# dic4['gender'].groups

# =================================================================
# Class_Ex1:
# From the data table above, create an index to return all rows for
# which the phylum name ends in "bacteria" and the value is greater than 1000.
# ----------------------------------------------------------------
import PyPDF2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.DataFrame({'value':[632, 1638, 569, 115, 433, 1130, 754, 555],
                     'patient':[1, 1, 1, 1, 2, 2, 2, 2],
                     'phylum':['Firmicutes', 'Proteobacteria', 'Actinobacteria',
    'Bacteroidetes', 'Firmicutes', 'Proteobacteria', 'Actinobacteria', 'Bacteroidetes']})


# data[data['value']>1000]
# data.loc[data['value']>1000]
#
# opt = ['bacteria']
#
# data[data['phylum'].isin(opt)]
#
# data.loc[data['phylym'].isin(opt)]

#new = data[data['phylum'].str.endswith('bacteria')]
#print(new)
#print(new[new['value']>1000])


print(data[(data['phylum'].str.endswith('bacteria')) & (data['value'] > 1000)])


print('#',50*"-")
# =================================================================
# Class_Ex2:
# Create a treatment column and add it to DataFrame that has 6 entries
# which the first 4 are zero and the 5 and 6 element are 1 the rest are NAN
# ----------------------------------------------------------------

data['treatment'] = np.zeros(8)
data['treatment'][5] = 1
data['treatment'][6] = 1
data['treatment'][7] = None

print(data)

print('#',50*"-")
# =================================================================
# Class_Ex3:
# Create a month column and add it to DataFrame. Just for month Jan.
# ----------------------------------------------------------------

data.insert(4,'month','Jan')
print(data)

print('#',50*"-")
# =================================================================
# Class_Ex4:
# Drop the month column.
# ----------------------------------------------------------------

data = data.drop(['month'],axis=1)
print(data)

print('#',50*"-")
# =================================================================
# Class_Ex5:
# Create a numpy array that has all the values of DataFrame.
# ----------------------------------------------------------------

print(data.to_numpy())


print('#',50*"-")

# =================================================================
# Class_Ex6:
# Read baseball data into a DataFrame and check the first and last
# 10 rows
# ----------------------------------------------------------------

df = pd.read_csv('baseball.csv')
print(df.head(10))
print(df.tail(10))

print('#',50*"-")
# =================================================================
# Class_Ex7:
# Create  a unique index by specifying the id column as the index
# Check the new df and verify it is unique
# ----------------------------------------------------------------

trial_copy2 = df.copy()
trial_copy2['new_id'] = trial_copy2['id']
new_id_c = trial_copy2.pop('new_id')
trial_copy2.insert(0,'new_id',new_id_c)
trial_copy2.set_index('id', inplace=True)
trial_copy = trial_copy2.copy()
print(' the index is unique',trial_copy.index.is_unique)
print(trial_copy)

# df.id.unique()
# unique = df.id.index.unique()
# df.insert(0,'id_index',unique)
# pd.Series(df['id']).is_unique
# pd.Series(df['id_index']).is_unique
# df
##duplicates on the column???
## dataframe without the duplicates??


print('#',50*"-")

# =================================================================
# Class_Ex8:
#Notice that the id index is not sequential. Say we wanted to populate
# the table with every id value.
# Hint: We could specify and index that is a sequence from the first
# to the last id numbers in the database, and Pandas would fill in the
#  missing data with NaN values:
# ----------------------------------------------------------------
#
# trial_copy.shape[0]
# number = 86641+trial_copy.shape[0]
# new_index = [e for e in range(86641,number)]
# trial_copy.reindex(new_index)
# trial_copy.insert(0,'id_index',new_index)
# trial_copy.set_index('id_index', inplace=True)
# trial_copy


trial_final = trial_copy.reindex(range(88641,89534))
print(trial_final)

#trial_copy.reindex(new_index)
# trial_copy.index ##bring  everything NAN
# reindex all the index of baseball
#sequence of orders
## SEQUENCE OF NUMBERS 88641
## RANGE REINDEX
#index or reindex list(range(88641:lastnumber+1))


print('#',50*"-")

# =================================================================
# Class_Ex9:
# Fill the missing values
# ----------------------------------------------------------------

trial_final = trial_final.fillna(method='ffill')
print(trial_final)
print('#',50*"-")

# =================================================================
# Class_Ex10:
# Find the shape of the new df
# ----------------------------------------------------------------

print('the shape is',trial_final.shape)

print('#',50*"-")

# =================================================================
# Class_Ex11:
# Drop row 89525 and 89526
# ----------------------------------------------------------------

# trial_final.drop('89525', axis = 1)
# trial_final.drop('89526', axis = 1)

trial_final = trial_final.drop(89525)
trial_final = trial_final.drop(89526)


print('#',50*"-")


# =================================================================
# Class_Ex12:
# Sor the df ascending and not descending
# ----------------------------------------------------------------
print(trial_final.sort_values('id'))
print('#',50*"-")

