### Ricardo Diaz
##EX 1

def array_algorithm(listy, target):
    n = len(listy)
    mini = 0
    maxi = n - 1
    while mini <= maxi: ## loop until the index are the same
        guess = (maxi + mini) // 2
        if listy[guess] == target:
            print("Your index is: ", end='')
            return guess
        elif target > listy[guess]: ##  listy guess provides the number and the index of the number
            mini = guess + 1 ##going to the right
        else:
            maxi = guess - 1 ###going to the left


sequence = [1, 3, 4, 5, 6, 7, 8, 9,13, 16]
target = 13


print(array_algorithm(sequence, target))
print("Cons, only works for list that are in sequence")
print("Pros, the code runs faster so you can find your target in an efficient way")
print("the process is call binary search, you find a midpoint of the sequence, \n"
      "if the target we are looking for is higher we choose the items on the right to the comparison and left is the target is lower, viceresa" )

## EX 2Work on a script to count the number of characters or (frequency) in a string.


new_list2 = 'ricardo'
d = dict()
for element in new_list2:
    d[element] = d.get(element,0) + 1 ##using dictionary method to get element value
    ##
print(d)




## EX 3Write a function that takes a list of words and returns the length of the longest one
def length_biggy(listy):
    new_list = []
    for letter in listy:
        new_list.append(len(letter))  ## aprending the the length of every word in list
    print("Lenght of the longest one is :",max(new_list)) ## getting the largest number in the new created list

trial = ['perro', 'gato', 'ricardo']

length_biggy(trial)

## EX 4Make up your own list and work on a program to get the smallest number from the list

n_list = [2,3,4,5,6,7,9]
print(min(n_list)) ##using min to get the mini number
## EX 5Work on a function that takes two lists and returns same (or True) if they have at least one
##common element

def compare(listy1,listy2):
    for l in listy1:
        for c in listy2: ##looping listy2 all elements for each element in list 1
            if c == l: ## same element breaks out of loop
                print('Same')
                return
        else:
            continue
listy1 = [1,3,4]
listy2 = [4,1,5]

compare(listy1,listy2)
## EX 6Work on a script to merge or join two dictionaries
x = {'a': 1, 'b': 2}
y = {'x': 3, 'c': 4}

def merge_dic(dic1,dict2):
    z = {**dic1,**dict2}
    return z

print(merge_dic(x,y))

## EX 7Work on a script or a program to map two lists into a dictionary.

listy1 = [1,3,4]
listy2 = ['a','b','c']

a  = dict(zip(listy1,listy2))
print(a)
## EX 8Answer all the class exercise questions and submit it (Check the instructions)

# =================================================================
# Class_Ex1:
# Write a program that simulates the rolling of a die.
# ----------------------------------------------------------------

from random import randrange,seed
while True:
    answer = input("Roll dice Y or N").upper()
    if answer == 'Y':
        a = randrange(1,7)
        if a == 1:
            print("+-----+")
            print("| *   |")
            print('|     |')
            print('|     |')
            print("+-----+")
        elif a ==2:
            print("+-----+")
            print("| * * |")
            print('|     |')
            print('|     |')
            print("+-----+")
        elif a ==3:
            print("+-----+")
            print("| * * |")
            print('| *   |')
            print('|     |')
            print("+-----+")
        elif a ==4:
            print("+-----+")
            print("| * * |")
            print('| * * |')
            print('|     |')
            print("+-----+")
        elif a ==5:
            print("+-----+")
            print("| * * |")
            print('| * * |')
            print('|  *  |')
            print("+-----+")
        else:
            print("+-----+")
            print("| * * |")
            print('| * * |')
            print('| * * |')
            print("+-----+")
    else:
        print("Bye")
        break

# =================================================================
# Class_Ex2:
# Answer  Ex1 by using functions.
# ----------------------------------------------------------------
from random import randrange,seed
def one():  ### printing patterns
    print("+-----+")
    print("| *   |")
    print('|     |')
    print('|     |')
    print("+-----+")
def two():
    print("+-----+")
    print("| * * |")
    print('|     |')
    print('|     |')
    print("+-----+")
def three():
    print("+-----+")
    print("| * * |")
    print('| *   |')
    print('|     |')
    print("+-----+")
def four():
    print("+-----+")
    print("| * * |")
    print('| * * |')
    print('|     |')
    print("+-----+")
def five():
    print("+-----+")
    print("| * * |")
    print('| * * |')
    print('|  *  |')
    print("+-----+")
def six():
    print("+-----+")
    print("| * * |")
    print('| * * |')
    print('| * * |')
    print("+-----+")

def dice(): ##making a function which generaets a random number an the print the dice
    a = randrange(1,7)
    if a == 1:
        one()
    elif a == 2:
        two()
    elif a == 3:
        three()
    elif a == 4:
        four()
    elif a == 5:
        five()
    else:
        six()


def playing_dice():
    x = 'Y'
    while x == 'Y':
        x = input("Roll dice Y or N: ").upper()
        if x == 'Y':
            dice()
        elif x == 'N':
            print('Thanks for playing')
        else:
            print('Please, use Y or N')
            playing_dice()

playing_dice()
# =================================================================
# Class_Ex3:
# Randomly Permuting a List
# ----------------------------------------------------------------
import random
random_list = [1,2,3,4]
##random_list2 = ['a','b','c','d']
random.shuffle(random_list)
print(random_list)



# =================================================================
# Class_Ex4:
# Write a program to convert a tuple to a string.
# ----------------------------------------------------------------
tp = (1,3,"word","pop",5)

nword = ""
for e in tp:
    nword += str(e)

print(nword)


# =================================================================
# Class_Ex5:
# Write a program to get the 3th element of a tuple.
# ----------------------------------------------------------------

print(tp)
print(tp[2:3])


# =================================================================
# Class_Ex6:
# Write a program to check if an element exists in a tuple or not.
# ----------------------------------------------------------------
tp = (1,3,"word","pop",5)

def check_tuple(thetuple,element):
    for e in thetuple:
        if e == element:
            print("Yeah it exists ")

check_tuple(tp,"word")



# =================================================================
# Class_Ex7:
# Write a  program to check a list is empty or not.
# ----------------------------------------------------------------

random_list = [1]
if len(random_list) == 0:
    print("list is empty")
else:
    print("list is not empty")


# =================================================================
# Class_Ex8:
# Write a program to generate a 4*5*3 3D array that each element is O.
# ----------------------------------------------------------------


listy_listy = [[[x for x in range(1)] for y in range(3)]for z in range(5)] ##list compheresions generating one line of 5*3 with only zeros
print('',listy_listy , '\n' ,listy_listy ,'\n' ,listy_listy ,'\n' , listy_listy) ##priting pattern




# l = [[[1],[2],[3]], [[3],[4],[5]]]
#
# new_listy = []
# for e in range(16):
#     new_listy.append([0])
# listy_listy = ([new_listy[0:3]]+[new_listy[3:6]]+[new_listy[6:9]]+[new_listy[9:12]]+[new_listy[12:16]])
# listy_listy2 = ([new_listy[0:3]]+[new_listy[3:6]]+[new_listy[6:9]]+[new_listy[9:12]]+[new_listy[12:16]])
# listy_listy3 = ([new_listy[0:3]]+[new_listy[3:6]]+[new_listy[6:9]]+[new_listy[9:12]]+[new_listy[12:16]])
# listy_listy4 = ([new_listy[0:3]]+[new_listy[3:6]]+[new_listy[6:9]]+[new_listy[9:12]]+[new_listy[12:16]])
# print('',listy_listy , '\n' ,listy_listy2 ,'\n' ,listy_listy3 ,'\n' , listy_listy4) STRUGGLING

# new_listy = []
# for e in range(16):
#     new_listy.append([0])
# print(new_listy)

##[[[number for number in group]] for group in new_listy]
  ##[[[a+1,b+1,c+1]] for [a,b,c] in new_listy]