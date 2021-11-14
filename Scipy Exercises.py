# =================================================================
# Class_Ex1:
# We will do some  manipulations on numpy arrays by importing some
# images of a racoon.
# scipy provides a 2D array of this image
# Plot the grey scale image of the racoon by using matplotlib
# ----------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
face = misc.face() ## Modify the face function
face.shape
plt.imshow(face)
plt.show()

#plt.imshow(face, cmap=plt.cm.gray)
#plt.imshow(face, cmap=plt.get_cmap('gray'))
# plt.show()
# red = face[:,:,0]
# plt.imshow(red, cmap='gray')
# plt.show()
# plt.imshow(face[:,:,1], cmap='gray',interpolation='none')
# plt.show()

type(face)

print('#',50*"-")
# =================================================================
# Class_Ex2:
# If still the face is gray choose the color map function and make it
# gray
# ----------------------------------------------------------------

plt.imshow(face[:,:,1], cmap='gray')
plt.show()



plt.imshow(face[:,:,1], cmap='gray') ## taking the image and maping grey
plt.show()






print('#',50*"-")
# =================================================================
# Class_Ex3:
# Crop the image (an array of the image) with a narrower centering
# Plot the crop image again.
# ----------------------------------------------------------------





center = face[150:450,400:800].copy()
plt.imshow(center)
plt.show()
center.shape


#raccon_arr = np.array(face)

face.shape
center = face[150:450,400:850].copy() #occupying array for only face in a new variable and plot it
center.shape # getting the shape of the raccon face so i can crop it out in the next exercise
plt.imshow(center)
plt.show()





print('#',50*"-")
# =================================================================
# Class_Ex4:
# Take the racoon face out and mask everything with black color.
# ----------------------------------------------------------------

import numpy as np
import scipy
import scipy.misc
import matplotlib.pyplot as plt

cum = face.copy()
cum[150:450,400:800] = np.zeros((300,400,3))
plt.imshow(cum)
plt.show()


trial = face.copy()
trial[150:450,400:850] = np.zeros((300,450,3))
plt.imshow(trial)
plt.show()


print('#',50*"-")
# =================================================================
# Class_Ex5:
# For linear equation systems on the matrix form Ax=b where A is
# a matrix and x,b are vectors use scipy to solve the for x.
# Create any matrix A and B (Size matters)
# ----------------------------------------------------------------
from scipy import linalg

a = np.matrix([[1,2,3,4,5],[1,2,4,5,3],[12,4,2,4,5],[12,4,2,4,5],[12,4,2,4,5]])
b = np.array([1,2,3,5,6])

linalg.solve(a,b)









a = np.matrix([[1,2,3],[4,5,6],[3,4,1]])
b = np.array([2,3,5])

x = linalg.solve(a,b)
print(x)
#B = np.matrix([[3,2,3],[4,5,8]])


print('#',50*"-")
# =================================================================
# Class_Ex6:
# Calculate eigenvalue of matrix A. (create any matrix and check your
# results.)
# ----------------------------------------------------------------


mat = np.random.randint(10,size=(2,2)) ##creating a random array
w,v = linalg.eig(mat)
print('E-value:', w)
print('E-vector', v)



a = np.array([[2, 2, 4],
              [1, 3, 5],
              [2, 3, 4]])
w,v=linalg.eig(a)
print('E-value:', w)
print('E-vector', v)



print('#',50*"-")
# =================================================================
# Class_Ex7:
# Sparse matrices are often useful in numerical simulations dealing
# with large datasets
# Convert sparse matrix to dense and vice versa
# ----------------------------------------------------------------
from scipy.sparse import csr_matrix

mat2 = np.random.randint(3,size=(4,4))
print('random array',mat2)
sparse_ma = csr_matrix(mat2)
print('sparse mattrix: ',sparse_ma)
dense_ma = sparse_ma.todense()
print('Dense matrix: ', dense_ma)
print('Dense matrix to Sparse matrix',csr_matrix(dense_ma))


print('#',50*"-")

# =================================================================
# Class_Ex8:
# Create any polynomial to order of 3 and write python function for it
# then use scipy to minimize the function (use Scipy)
# ----------------------------------------------------------------
from scipy.optimize import minimize
from numpy import poly1d
import random
import scipy.optimize as spo
# p = poly1d([3,6,9])
# print(p)

def poly():
    listy = []
    while len(listy) <= 2:
        for num in range(3):
            num = random.choice([1,2,3,4,5,6,7,8,9,10])
            listy.append(num)
    arr = np.array(listy)
    return poly1d(arr)
poly()

# p = poly1d([3,6,9])
# print(p)

result = spo.minimize(poly(),1)
print(result)






print('#',50*"-")
# =================================================================
# Class_Ex9:
# use the brent or fminbound functions for optimization and try again.
# (use Scipy)
# ----------------------------------------------------------------

#Using the Brent method, we find the local minimum as:

#https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize_scalar.html#scipy.optimize.minimize_scalar
from scipy.optimize import minimize_scalar

result = minimize_scalar(poly()) ## brent
print('Brent method ', result)

# from scipy.optimize import minimize_scalar
# res = minimize_scalar(f)
# res.x
# 1.28077640403 ## this is the Brent Method
print('#',50*"-")
# =================================================================
# Class_Ex10:
# Find a solution to a function. f(x)=0 use the fsolve (use Scipy)
# ----------------------------------------------------------------

from scipy.optimize import fsolve


def f(x):
     y = 2.0 * x**2 + 3.0*x - 10.0
     return y

x = fsolve(f,3.0)
print(x)

yes = poly1d([2,4,6],variable='x')
fsolve(yes,3)




print('#',50*"-")
# =================================================================
# Class_Ex11:
# Create a sine or cosine function with a big step size. Use scipy to
# interpolate between each data points. Use different interpolations.
# plot the results (use Scipy)
# ----------------------------------------------------------------

from scipy.interpolate import interp1d
from scipy.interpolate import krogh_interpolate
from scipy.interpolate import barycentric_interpolate
from scipy.interpolate import pchip_interpolate



x = np.linspace(0,20,21)
y = np.sin(x)
#xnew = np.linspace(0,20,50)
# f = interp1d(x, y)
# f2 = interp1d(x,y, kind = 'quadratic')
trial_x = np.linspace(min(x), max(x), num=100)
trial_y = krogh_interpolate(x,y,trial_x) ##f3
trial_y3 = pchip_interpolate(x,y,trial_x)
trial_y2 = barycentric_interpolate(x,y,trial_x)
plt.plot(trial_x,trial_y,'--',label = 'Krog Interporlation')
plt.plot(trial_x,trial_y3, label ='Pchip Interpolation')
plt.plot(trial_x,trial_y2, label ='Barycentric Interporlation')
plt.legend()
plt.show()




##type of interpolation





print('#',50*"-")
# =================================================================
# Class_Ex12:
# Use scipy statistics methods on randomly created array (use Scipy)
# PDF, CDF (CUMsum), Mean, Std, Histogram
# ----------------------------------------------------------------
import numpy as np
from scipy.stats import gaussian_kde
from scipy.stats import norm
from scipy.stats import ttest_1samp


arr = norm.rvs(size=100)  #random generation
mean = arr.mean()
std = arr.std()
cum_sum = arr.cumsum()
print('std: ',std,'mean: ', mean,'\ncum_sum: ',cum_sum)
pdf = norm.pdf(arr,mean,std)
plt.plot(arr,pdf) ##change x
plt.title('Array PDF')
plt.show()
cdf = norm.cdf(arr,mean,std)
plt.plot(arr,cdf)
plt.title('Array CDF')
plt.show()
plt.hist(arr,density= True)
plt.title('Array Histogram')
plt.show()


#arr = np.random.randint(100,size = (1000,))
#arr = np.linspace(-5,5,1000)
# x=numpy.linspace(dataDiff.min(),dataDiff.max(),1000)
# pdf=norm.pdf(x,mean,std)
# plt.plot(x,pdf)
# data = numpy.array([[113,105,130,101,138,118,87,116,75,96,
#              122,103,116,107,118,103,111,104,111,89,78,100,89,85,88],
#          [137,105,133,108,115,170,103,145,78,107,
#               84,148,147,87,166,146,123,135,112,93,76,116,78,101,123]])
# dataDiff = data[1,:]-data[0,:]
# dataDiff.mean(), dataDiff.std()
# plt.rcParams['figure.figsize'] = (15.0, 5.0)
# plt.hist(dataDiff)
# plt.show()



print('#',50*"-")
# =================================================================
# Class_Ex13:
# USe hypothesise testing  if two datasets of (independent) random varibales
# comes from the same distribution (use Scipy)
# Calculate p values.
# ----------------------------------------------------------------
from scipy.stats import ttest_ind
arr2 = norm.rvs(loc = 50,scale = 60, size=1000)
arr3 = norm.rvs(loc = 50, scale =60,size=1000)
print('Ho arr 2 is independent of arr 3')
print('H1 arr 2 is dependent of arr 3 ')
print(ttest_ind(arr2,arr3))




print('#',50*"-")
# ----------------------------------------------------------------