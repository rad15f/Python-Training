## RICARDO DIAZ HW3
# =================================================================
# Class_Ex1:
# Writes a python script (use class) to simulate a Stopwatch .
# push a button to start the clock (call the start method), push a button
# to stop the clock (call the stop method), and then read the elapsed time
# (use the result of the elapsed method).
# ----------------------------------------------------------------

import time


class Stopwatch:
    def __init__(self):
        self.start = 0
        self.elapsed = 0

    def start_method(self):

        self.start = time.time()

    def stop_method(self):
        self.elapsed_method()


    def elapsed_method(self):
        self.elapsed = time.time() - self.start
        print("Elapsed Time:", self.elapsed)

# =================================================================
# Class_Ex2:
# Write a python script (use class)to implement pow(x, n).
# ----------------------------------------------------------------


class Powy():
    def __init__(self,x,n):
        self.x = x
        self.n = n
    def implement_pow(self):
        return self.x**self.n





# =================================================================
# Class_Ex3:
# Write a python class to calculate the area of rectangle by length
# and width and a method which will compute the area of a rectangle.
# ----------------------------------------------------------------


class Rectangule:
    def __init__(self, length, width): #provide lenght and width
        self.length = length
        self.width = width

    def rectangle_area(self):
        print("Your rectangule area: ")
        return self.width * self.length

# =================================================================
# Class_Ex4:
# Write a python class and name it Circle to calculate the area of circle
# by a radius and two methods which will compute the area and the perimeter
# of a circle.
# ----------------------------------------------------------------
import math

class Circle:
    def __init__(self, radius):
        self.radius = radius ## provide radius

    def area_circle(self):
        return math.pi * self.radius ** 2

    def perimeter_circle(self):
        return 2 * math.pi * self.radius

# HW3
# E.1:
# Write a script to find duplicates from an array (define an array with some duplicates on it). If
# you use built in function in python explain the methods and how this methods are working
listy = [1,4,6,7,8,9,'safa','a','b','c',2,4,6,7,'a','c']

for element in listy:
    if listy.count(element) >= 2: ## this another method which count the same elemnts of a list
        print("There is duplicates!")
        break
else:
    print('There are no duplicates')
#E2:
#Write a script that finds all such numbers which are divisible by 2 and 5, less than 1000. If you
#use built in function in python explain the methods and how this methods are working.

def divisible():
    new_list = []
    for number in range(1,1001):
        if number % 2 == 0 and number % 5 == 0:
            new_list.append(number) ## Append put a element at the end of the list. This append method refers to a function which is part of a class
        else:
            continue
    return new_list

#Ex3
#Write a Python class to convert a roman numeral to an integer. Hint: (Use the following symbols
#and numerals

class Roman:
    def __init__(self,word):
        self.word = word
        self.new_listy = []
        self.number = 0

    def transform(self):
        roman = self.word
        roman_list = list(roman)
        roman_list.append('0') ## appending 0 cause my loop wont count the last element... nasty
        i = 0
        n = len(roman)
        while i < n:
            if roman_list[i] + roman_list[i+1] == 'CM': ##checking for CM , CD ,XC ,etc and put it together to the new_list
                self.new_listy.append('CM')                # this all are exceptions
                i = i + 2
            elif roman_list[i] + roman_list[i+1] == 'CD':
                self.new_listy.append('CD')
                i = i + 2
            elif roman_list[i] + roman_list[i+1] == 'XC':
                self.new_listy.append('XC')
                i = i + 2
            elif roman_list[i] + roman_list[i+1] == 'XL':
                self.new_listy.append('XL')
                i = i + 2
            elif roman_list[i] + roman_list[i+1] == 'IX':
                self.new_listy.append('IX')
                i = i + 2
            elif roman_list[i] + roman_list[i+1] == 'IV':
                self.new_listy.append('IV')
                i = i + 2
            else:
                self.new_listy.append(roman_list[i])
                i += 1

        d = {'I': 1, 'IV': 4, 'V': 5, 'IX': 9, 'X': 10, 'XL': 40, 'L': 50, 'XC': 90, 'C': 100, 'CD': 400, 'D': 500, 'CM': 900, 'M': 1000}
        n = 0
        for e in self.new_listy: ## new list of roman numbers
            if e in d.keys(): ##check if the roman number is on the dic and provide the value
                n= d[e] + n # adding the value
        return n


trial = Roman('MMCDLXXIV')
trial.transform()


##E.4:
# Write a Python class to find sum the three elements of the given array to zero.
# Given: [-20, -10, -6, -4, 3, 4, 7, 10]
# Output : [[-10, 3, 7], [-6, -4, 10]]


class Sum:

    def __init__(self, listy):
        self.listy = listy
        self.len = len(listy)

    def find_sum(self):
        new = []
        for x in range(0, self.len - 2):
            for y in range(x+1, self.len - 1):
                for z in range(y+1, self.len): ##using three for loops so we use iteration to see any possible combination
                    if (self.listy[x] + self.listy[y] + self.listy[z] == 0): ## always trying different number so we dont get the same twice thats why we use diferent ranges
                        new.append([listy[x], listy[y], listy[z]]) ##appending a list to a list
        return new

