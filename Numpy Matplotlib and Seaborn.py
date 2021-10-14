#HOMEWORK 5
# E.1: Write a script to i. Sum all the items in a array (use random number generator and multiply it by 100, create a vector with the size 200).
# ii. Get the largest number and smallest number with the indexing of it. iii. plot the following vector
# and check your min and max value that you find in section i.
import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np

arr = np.array((100*np.random.random(200)))
print(arr)
sumy = np.sum(arr)
print('Sum of all items', sumy)
miny = np.amin(arr)
index_miny = np.where(arr == miny)
print('Minimun Value: ', miny, 'Index: ', index_miny)
maxy= np.amax(arr)
index_maxy = np.where(arr == maxy)
print('Maximun Value:  ', maxy, 'Index: ', index_maxy)

plt.plot(arr)
plt.axhline(maxy)
plt.axhline(miny)
plt.show()

# E.2:
# Plot the following functions x, sin(x), e**x
# and log(x) over the interval 1 < x < 6 with the step
# size of 0.1. (Put the title x axis label and y axis label for each plot)

x = np.linspace(1,6,60)

y = [x, np.sin(x),np.exp(x),np.log(x)]

t = ['Plot X', 'Plot Sin(x)','Plot Exp(x)', 'Log X']

n =0
for e in y:
    plt.plot(x,e)
    plt.title(label = t[n])
    plt.xlabel('X Asis')
    plt.ylabel('Y Asis')
    plt.show()
    n += 1

#E.3:
# Generate the random gaussian numbers with zero mean and variance of 1 called it vector x,
# generate the random uniform numbers with zero mean and variance of 1 called it vector y .
# i. Compute the mean and standard deviation of x and y.
# ii. Plot the histogram of x and y, increase the number of bins to get more resulting. Explain what
# information you get from the histogram(Put the title x axis label and y axis label for each plot)
#

import random
x = []
for l in range(500):
    x.append(random.gauss(0,1))
x = np.array(x)

y = []
for l in range(500):
    y.append(random.uniform(-1,1)) ## CREATE NUMBER FROM -1 0 1
y = np.array(y)
type(y)

fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
plt.hist(x, bins = 50)
plt.title(' X axis label')
ax2 = fig.add_subplot(2,2,2)
plt.hist(y,bins = 50)
plt.title(' Y axis label')
plt.show()

print('Random Gaus histogram shows a normal distribution amd Random Uniform histogram is more equally distributed with a variance of 1 ')
# E.4:
# Lets
# A =
#
# 1 2 3
# 4 5 6
# 7 8 9
#
# Answer the following questions (Do not put the digits manually ):
# i. Assign vector x to the first row of A.
# ii. Assign matrix y to the last 2 rows of A
# iii. Sum the first row and add it to the first column.
# iv. Compute the norm of x (Euclidian Norm).
# v. Swap the first column with the second column and delate the second row.

arr = np.arange(1,10).reshape(3,3)
x = arr[:,0]
print(x)
y = arr[:,1:3]
print(y)
z = np.sum(x)
print('Sum the first row,',z)

arr[:,0] = arr[:,0] + 12
print('Sum the first row and add it to the first column,',arr)
np.linalg.norm(arr)

arr[:,0],arr[:,1] = arr[:,1],arr[:,0].copy()

# E5
# i. Create a vector between 20 and 35, square each elements and sum all the elements of this
# vector.
# Let
# x =
# 2 −4 9 −8
# −3 2 −1 0
# 5 4 −3 3
# ii. Compute the absolute value of x for all the rows and columns separately.
# iii. Compute the square of each elements of x.
# iv. Swap the first row by the second row.
# v. Replace the first row by zeros and the third row by ones.
# vi. Compute the mean and standard deviation of first and third row.
# vii. Sum all the columns and then sum the results.

arr = np.arange(20,36)
print(arr)
type(arr)
arr2 = np.square(arr)
print('Squaring each elementl',arr2)
print('Sum all the elements,',np.sum(arr2))

arr = np.array([[2,-4,9,-8],[-3,2,-1,0],[5,4,-3,3]])
print(arr)
print('Firt Column',np.abs(arr[:,0]))
print('Second Column',np.abs(arr[:,1]))
print('Third Column',np.abs(arr[:,2]))
print('Four Column',np.abs(arr[:,3]))
print('First Row',np.abs(arr[0,:]))
print('Second Row',np.abs(arr[1,:]))
print('Third Row',np.abs(arr[2,:]))
print(arr)
print(np.square(arr))
##iv. Swap the first row by the second row.
arr[0,:],arr[1,:]= arr[1,:],arr[0,:].copy()
print(arr)

arr[0,:] = np.zeros(4)
arr[2,:] = np.ones(4)
print(arr)



#vi. Compute the mean and standard deviation of first and third row.
print('First row mean',np.mean(arr[0,:]))
print('Second row mean',np.mean(arr[2,:]))
print('Standard Deviation first row', np.std(arr[0,:]))
print('Standard Deviation third row', np.std(arr[2,:]))

#vii. Sum all the columns and then sum the results.
colum_sum = arr.sum(axis =0 )
print('All columns sum',colum_sum)

print('Sum the results', np.sum(colum_sum))



#Exercises 6
languages = ['Java', 'Python', 'PHP', 'JavaScript', 'C', 'C++']
usage = [22.2, 17.6, 8.8, 8, 7.7, 6.7]
plt.bar(languages, usage)
plt.ylabel('Usage percentage')
plt.title('Languages vs percentage')
plt.show()

#E.7:
# Write a Python code to create bar plots with errorbars on the same plot. Put labels above each
# bar displaying men average (integer value).
# Sample Date
# Average: 0.14, 0.32, 0.47, 0.38
# STD: 0.23, 0.32, 0.18, 0.46

average = np.array([ 0.14, 0.32, 0.47, 0.38])
std = np.array([0.23, 0.32, 0.18, 0.46])

plt.bar(average,std, width = 0.04, linewidth = 0.5, alpha=0.5)
plt.errorbar(average,std,fmt ='o',color='r')

for x,y in enumerate(std):
    plt.text(average[x],y,str(y)) ## putting the position x ,y plus the string
plt.show()

# E.8:
# Write a script to find the second largest number in an array (use random number generator) and
# multiply it by 50.

t = np.random.randint(20,size=(10))
print(t)

sorted1 = np.sort(t)
print('Second largest: ',sorted1[len(sorted1)-2])

print('#',50*"-")

### CLASS EXERCISE 1
# =================================================================
# Class_Ex1:
# Write a NumPy code to test if none of the elements of a given
# array is zero.
# ----------------------------------------------------------------
import numpy
import numpy as np

x = np.array([1,2,3,4,5,6,7,8])

np.all(x) ## true if no  zero
# false if zero




# =================================================================
# Class_Ex2:
# Write a NumPy code to test if none of the elements of a given
# array is non-zero.
# ----------------------------------------------------------------

y = np.array([1,2,3,4,5,6,0,8,9])

for e in y:
    if np.all(e) == False:
        print('There are zeros')
        break
else:
    print('There are no zeros')

##np.nonzero(y)
##np.any(y)




# =================================================================
# Class_Ex3:
# Write a NumPy code to test if two arrays are element-wise equal
# within a tolerance.
# ----------------------------------------------------------------

##np.array_equal(x,y) ##checking it two are equal
x = np.array([5e5, 1e-7, 4.000004e6])
y = np.array([5.00001e5, 1e-7, 4e6])

print(np.allclose(x,y))



# =================================================================
# Class_Ex4:
# Write a NumPy code to create an array with the values
# 1, 8, 130, 10990005 and determine the size of the memory occupied
# by the array.
# ----------------------------------------------------------------

arr = np.array([1,8,130,10990005])
print(arr.nbytes)




# =================================================================
# Class_Ex5:
# Write a NumPy code to create a array with values ranging from
# 10 to 20 and print all values except the first and last.
# ----------------------------------------------------------------

arr = np.arange(11,20)
print(arr)


# =================================================================
# Class_Ex6:
# Write a NumPy code to reverse (flip) an array (first element becomes last).
# ----------------------------------------------------------------

arr = np.arange(1,20)
arr2 = arr[::-1]
print(arr2)





# =================================================================
# Class_Ex7:
# Write a NumPy code to create a matrix with 1 on the border and 0 inside.
# ----------------------------------------------------------------

trial = np.ones((5,5))
trial2 = np.zeros((3,3))

##print(trial)
trial[1:4,1:4] = trial2
print(trial)




# =================================================================
# Class_Ex8:
# Write a NumPy code to add a border (filled with 0's) around a 3x3
# matrix of one.
# ----------------------------------------------------------------

trial = np.zeros((5,5))
trial2 = np.ones((3,3))

##print(trial)
trial[1:4,1:4] = trial2
print(trial)





# =================================================================
# Class_Ex9:
# Write a NumPy code to append values to the end of an array.
# ----------------------------------------------------------------

arr = np.array([1,2,3])
arr2 = np.array([4,3,2])

np.append(arr,arr2)



# =================================================================
# Class_E10:
# Write a NumPy code to find the set difference of two arrays.
# The set difference will return the sorted, unique values in array1
# that are not in array2.
# ----------------------------------------------------------------

arr = np.array([5,3,2,1,7])
arr2 = np.array([2,3,4,6,8])


diff = np.sort(np.setdiff1d(arr,arr2))
print(diff)



# =================================================================
# Class_Ex11:
# Write a NumPy code to construct an array by repeating.
# Sample array: [1, 2, 3, 4, 5]
# ----------------------------------------------------------------
import numpy as np

arr = np.array([1,2,3,4,5])

t = np.tile(arr,3)
print(t)

##print(np.repeat([1,2,3,4],3))

# =================================================================
# Class_Ex12:
# Write a NumPy code to get the values and indices of the elements
# that are bigger than 6 in a given array.
# ----------------------------------------------------------------

a = np.arange(10).reshape(2,5)
print("Value are bigger that 6: ", end='')
print(a[a>6])
# new_a = [a>6]
# t_index = np.where(new_a)
# print(t_index)
# print(a)
print('There is indices are ', np.nonzero(a>6))
# np.where(a>6)
#

# =================================================================
# Class_Ex13:
# Write a NumPy program to find the 4th element of a 2 dimensional
# specified array.
# ----------------------------------------------------------------

a = np.arange(16).reshape(4,4)
type(a)
print("Array:",a)
print("Four element",a[0,3])

a.ndim



# =================================================================
# Class_Ex14:
# Write a NumPy code to get the floor, ceiling and truncated
# values of the elements of an numpy array.
# ----------------------------------------------------------------
import numpy as np
t = np.array([1.4,2.2,2.5,3.1,3.6])

print(np.floor(t)) ## normal round
print(np.ceil(t)) ## always round up no matter what
print(np.trunc(t)) ## always round down

# =================================================================
# Class_Ex15:
# Write a NumPy code to compute the factor of a given array by
# Singular Value Decomposition.
# ----------------------------------------------------------------
import numpy as np
arr = np.array([[0,0,0,0,1],[0,2,2,0,1],[1,0,0,0,2],[0,2,1,2,1]])
print(arr)

u,s,v = np.linalg.svd(arr,full_matrices=False)

print('U:',u,'\nS:',s,'\nV:',v)






# =================================================================
# Class_Ex16:
# ----------------------------------------------------------------
# Write a NumPy code to compute the eigenvalues and right eigenvectors
# of a given square array.
# ----------------------------------------------------------------

arr = np.arange(4).reshape(2,2)
print(arr)

eigen, righeigen = np.linalg.eig(arr)

print('Eigen value:', eigen)
print('Right eigenvector', righeigen )

# =================================================================
# Class_Ex17:
# Write a NumPy code to get the dates of yesterday, today and tomorrow.
# ----------------------------------------------------------------

y = np.datetime64('today') - np.timedelta64(1,'D')
t = np.datetime64('today')
to = np.datetime64('today') + np.timedelta64(1,'D')
print(y,t,to)


# =================================================================
# Class_Ex18:
# Write a NumPy code to find the first Monday in June 2021.
# ----------------------------------------------------------------

datey = '2021-06'

date = np.busday_offset(datey, 0, roll='forward',weekmask='Mon')
print(date)


# =================================================================
# Class_Ex19:
# Write a NumPy code to find the roots of the following polynomials.
# a) x2 − 3x + 8.
# b) x4 − x3 + -x2 + 1x – 2
# ----------------------------------------------------------------

a = np.poly1d([2,-3,8], variable='x')
print(a)

b = np.poly1d([4,-3,-2,1,-2],variable='x')
print(b)

print('A Polynomial roots',np.roots(a))
print('B Polynomial roots',np.roots(b))


# =================================================================
# Class_Ex20:
# Write a NumPy program to calculate mean across dimension, of matrix.
# ----------------------------------------------------------------

arr = np.array([[5,8],[2,6]])
print(arr)

mean_col = arr.mean(axis=0)
mean_row = arr.mean(axis=1)
print(mean_col,mean_row)

print('#',50*"-")

# =================================================================
# Class_Ex1:
# Class_Ex1:
# Find the slope of the following curve for each of its points
#                  y = np.exp(-x ** 2)
# Then plot it with the original curve np.exp(-X ** 2) in the range
# (-3, 3) with 100 points in the range
# ----------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-3,3,100)
y = np.exp(-x **2)
plt.show()
m, b = np.polyfit(x, y,1) #slope
plt.plot(x,y)
plt.plot(x, m*x + b)
plt.show()





# =================================================================
# Class_Ex2:
# A file contains N columns of values, describing N–1 curves.
# The first column contains the  x coordinates, the second column
# contains the y coordinates of the first curve, the third
# column contains the y coordinates of the second curve, and so on.
#  We want to display those N–1 curves.

# ----------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('data2.txt', dtype= int)  #obtaining the data
print(data)
x = data[:, 0] #creating constant x
#data.shape
print(data.shape[1])
for e in range(1,data.shape[1]): ## range 2. start with 1 since the first column is constant (x)
        y = data[:,e] #plotting y with column index 1 and the plotting y with column index 2
        plt.plot(x,y)
plt.title('Data Line Graph')
plt.show()




# =================================================================
# Class_Ex3:
# Write a efficient code to stack any number of layers of data into
# a bar chart plot.
# Use the following data.
# ----------------------------------------------------------------

#___________________________________________________________________________
data = np.random.rand(5,3)
print(data)
color_list = ['b', 'g', 'r', 'k', 'y']

#print(data.shape)
#print(np.arange(5))

#data.shape

for i in np.arange(data.shape[0]):
    plt.bar([0,1,2],data[i],color=color_list[i]) ##itirating thru colors and data
    ## x is static , 3 positions.
plt.title("Random Bar Plot")
plt.show()
print('\n')

# xpos = np.arange(len(color_list))
# print(xpos)
# plt.bar(xpos,data)
# plt.xticks(xpos,data)
# plt.show()
# =================================================================
# Class_Ex4:
# Write a Python code to plot couple of lines
# on same plot with suitable legends of each line.
# ----------------------------------------------------------------


x1 = [10,20,30,40]
x2 = [20,30,20,30]
y1 = [30,20,10,20]
y2 = [30,20,30,10]

plt.xlabel('x -axis')
plt.ylabel('y -axis')

plt.plot(x1,y1,label = 'X1 Y1')
plt.plot(x2,y2, label = 'X2 Y2')
plt.legend()
plt.show()






# =================================================================
# Class_Ex5:
# Write a Python code to plot two or more lines with legends,
# different widths and colors.
# ----------------------------------------------------------------



x1 = [10,20,30,40]
x2 = [20,30,20,30]
y1 = [30,20,10,20]
y2 = [30,20,30,10]

plt.xlabel('x -axis')
plt.ylabel('y -axis')

plt.plot(x1,y1, color = 'b', linewidth = 1.0, label = 'X1 Y1')
plt.plot(x2,y2, color = 'r', linewidth = 2.0, label = 'X2 Y2 ')
plt.legend()
plt.show()








# =================================================================
# Class_Ex6:
# Write a Python code to plot two or more lines and set the line markers.
# ----------------------------------------------------------------


x1 = [10,20,30,40]
x2 = [20,30,20,30]
y1 = [30,20,10,20]
y2 = [30,20,30,10]

plt.xlabel('x -axis')
plt.ylabel('y -axis')


plt.plot(x1,y1, color = 'b', linewidth = 1.0, label = 'line1', linestyle = 'dashdot')
plt.plot(x2,y2, color = 'r', linewidth = 1.0, label = 'line2', linestyle = 'dashdot')
plt.legend()
plt.show()





# =================================================================
# Class_Ex7:
# Write a Python code to show grid and draw line graph of
# revenue of certain compan between November 4, 2017 to November 4, 2018.
# Customized the grid lines with linestyle -, width .6. and color blue.
# ----------------------------------------------------------------

import datetime
from matplotlib import pyplot as plt
from matplotlib.dates import date2num

data = [(datetime.datetime.strptime('2017-11-04', "%Y-%m-%d"), 100), #setting up my dates with values for my y
        (datetime.datetime.strptime('2018-11-04', "%Y-%m-%d"), 500)] ## artificial 100 500 revenues for those dates

x = [date2num(date) for (date, value) in data]  # creating a array of dates values
y = [value for (date, value) in data]  ##  values 100 500 to x
fig = plt.figure()
ax = fig.add_subplot()
ax.plot(x,y,'purple')
plt.grid(color = 'blue',linestyle ='-',linewidth =0.5)
ax.set_xticks(x)
ax.set_xticklabels(
        [date.strftime("%Y-%m-%d") for (date, value) in data]
        ) ## showing the dates on my x
plt.xlabel('Dates')
plt.ylabel('Revenue')
plt.title('Ricardo Company')

plt.show()


# =================================================================
# Class_Ex8:
# Write a Python code to create multiple empty plots  in one plot
# (facets)
# ----------------------------------------------------------------



plt.subplots(nrows=3, ncols=3) ##printing 3 empty plots
plt.tight_layout() #order it
plt.show()

print('#',50*'-')

# =================================================================
# Class_Ex1:
# We will be working with a famous titanic data set for these exercises.
# Later on in the Data mining section of the course, we will work  this data,
# and use it to predict survival rates of passengers.
# For now, we'll just focus on the visualization of the data with seaborn:

# use seaboran to load dataset
# ----------------------------------------------------------------
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
sns.set_style('whitegrid')
titanic = sns.load_dataset('titanic')

# =================================================================
# Class_Ex2:
# Join plot on fare and age
# ----------------------------------------------------------------
print(' SEABORN GRAPHS TITANIC ANALYSIS')

titanic.head()

sns.jointplot(x='fare', y='age', data = titanic)
plt.show()

print('You can see here where the majority of the customers density',
      'and how does the age and fare relate')

# =================================================================
# Class_Ex3:
# Distribution plot on fare with red color and 35 bin
# ----------------------------------------------------------------

sns.displot(titanic['fare'] , color = 'red' , bins = 35)
plt.show()
print('In the distribution plot you can see most of the customer went to the cheaper fare')


# =================================================================
# Class_Ex4:
# box plot on class and age
# ----------------------------------------------------------------

sns.boxplot(x='class', y = 'age', data = titanic)
plt.show()
print('Here you can see a boxplot age vs class, demostrating the range, mean and identifies our outliers')

# =================================================================
# Class_Ex5:
# swarmplot on class and age
# ----------------------------------------------------------------

sns.swarmplot(x='class' ,y = 'age', data = titanic)
plt.show()
print('Another way to see age vs class, identity outliers , density. Its better to use boxplot in this case')


# =================================================================
# Class_Ex6:
# Count plot on sex
# ----------------------------------------------------------------

sns.countplot(titanic['sex'])
plt.show()

print('Counting the amount of customers separating sexes')



# =================================================================
# Class_Ex7:
# plot heatmap
# ----------------------------------------------------------------



sns.heatmap(titanic.corr())
plt.show()
print('A heatmap showing the correlation between different columns in the dataset')

# =================================================================
# Class_Ex8:
# Distribution of male and female ages in same grapgh (Facet)
# ----------------------------------------------------------------

g = sns.FacetGrid(titanic, col='sex')
g.map(sns.distplot, 'age')
plt.show()

print('This graph shows normal distribution on age separating on sexes, the majority of',
      'the customers were around 30 in the males and femala 20-30')

# =================================================================
# Class_Ex9:
# Explain each graph and describe the results in words
# ----------------------------------------------------------------
