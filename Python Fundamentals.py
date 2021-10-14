 # =================================================================
# Class_Ex1: 
# Write python program that converts seconds to 
# (x Hour, x min, x seconds)
# ----------------------------------------------------------------
## seconds = int(input("Put your seconds here:"))

#second = int(input("Put your seconds here: "))

def conversion(second):
    hour = round(second/3600,2)
    min = second/60
    sec = second
    print(hour , " Hour", min , " min", sec , " seconds" )

#conversion(120)



##pint(seconds/3600 + "Hour," + seconds/60 + "min," + seconds)r


# =================================================================
# Class_Ex2: 
# Write a python program to print all the different arrangements of the
# letters A, B, and C. Each string printed is a permutation of ABC.
# ----------------------------------------------------------------
def permutations(word):
    if len(word) == 1:##check this shit
        return [word]

    perms = permutations(word[1:]) #all possible perms -1
    choosen_letter = word[0]
    new_perm_list = []

    for perm in perms:
        for i in range(len(perm)+1):
            new_perm_list.append(perm[:i] + choosen_letter + perm[i]) # placing our char in any possible position in our word
    return new_perm_list

print(permutations('abc'))

def all_perms(elements):
    if len(elements) <=1:
        yield elements
    else:
        for perm in all_perms(elements[1:]):
            for i in range(len(elements)):
                # nb elements[0:1] works in both string and list contexts
                yield perm[:i] + elements[0:1] + perm[i:]



def all_perms2(s):
    if len(s) <= 1:
        yield s
    else:
        for i in range(len(s)):
            for p in permutations(s[:i] + s[i+1:]):
                yield s[i] + p


def perm1(lst):
    if len(lst) == 0:
        return []
    elif len(lst) == 1:
        return [lst]



# =================================================================
# Class_Ex3: 
# Write a python program to print all the different arrangements of the
# letters A, B, C and D. Each string printed is a permutation of ABCD.
# ----------------------------------------------------------------





# =================================================================
# Class_Ex4: 
# Suppose we wish to draw a triangular tree, and its height is provided 
# by the user.
# ----------------------------------------------------------------

height = 12
tree = []
for n in range(height):
    tree.append("   *")
    print(("\t")*(height-n), tree)

for i in range(0,height): # selecting how many ROWS/HEIGHT do you want
    for x in range(0,height-i): # nested for loop to add spaces in your row
        print(" ", end = "")
    for y in range(1): #  nested for loop to add 0 in your rows
        print(" 0"* i)


# =================================================================
# Class_Ex5: 
# Write python program to print prime numbers up to a specified values.
# ----------------------------------------------------------------
number = 23
prime_list = []

for x in range(2,number+1):
    for n in range(2,x):
        if (x % n) == 0:
            break
    else:
        prime_list.append(x)

print(prime_list)



# =================================================================