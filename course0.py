#This is the basics course 0
#here I will teach you the very basics of python
#in the course we will be using spyder to make a smooth transition
#for the people coming from matlab
#try executing this line by line or block by block and think about what the code is doing
#play around and start to get creative

#getting help
help(5)

#initializing variables
a = 3
a += 3
a -= 3
#%%
#print them
print(a)

#strings can be added just like numbers
string = 'Hallo'
string += ' Welt!'

print(string)

#switch a and b values and pass tuples
a = 23
b = 42
b, a = a, b

#lists are very powerful... they organize numbers as lists!

list = [1,1,2,3,5,8,13,42]

#dicts are kind of like a list but store things not sorted
#but rather as key-value pairs ... kind of like a dictionary
mydict = {"Key 1": "Value 1", 2: 3, "pi": 3.14}

print(mydict['pi'])
print(mydict['Key 1'])
print(mydict[2])
#this would create an error because there is no key named 0
#print(mydict[0])

#yoiu can make lists of things other than numbers
sample = [1, ["another", "list"], ("a", "tuple")]
mylist = ["List item 1", 2, 3.14]
mylist[0] = "List item 1 again" # We're changing the item.
mylist[-1] = 3.21 # Here, we refer to the last item.

#create an empty list or dict to add things later
my_list = []
my_dict = {}

#add entries to the empty list
my_list.append(1)

#you can even append a list to a list ...
my_list.append(['a','b'])
my_list.append(['a1','b1'])

#now list contains two lists
print(my_list)

#only print the first list
print(my_list[1][0])
print(my_list[1])

#the best library is numpy for anythign with numbers ...
import numpy as np

#is this ma matrix?
data = np.array([[1, 1.1, 1.2],
                 [2.1, 2.1, 2.6],
                 [1.2, 5.2, 8.44],
                 [5.6, 7.4, 5.45],
                 [3.8, 3.8, 2.32]])
print(data)
#you can print slices through a matrix!!
print(data[:,1])
print(data[1,:])

#sometimes you need to print things in a formatted way
print("X: %s mm Y: %s mm Z: %s mm" % (42, 23, 0.01))
print("X: {} mm Y: {} mm Z: {} mm".format(42, 23, 0.01))

print("This %(verb)s a %(noun)s." % {"noun": "test", "verb": "is"})

#a little bit of randomness is sometimes good
from random import randint as zufallsInt
zufallsZahl = zufallsInt(1,5000)
print(zufallsZahl)



#looping in python works like this
#you loop with something over an iteratable
for a in [1,2,3,4]:
  print(a)

 #range essentially gives an iterable to do something 10 times ... starts at 0!!

rangelist = range(10)
print(rangelist)

#a more complex example:
for number in rangelist:
    # Check if number is one of
    # the numbers in the tuple.
    if number in (3, 4, 7, 9):
        # "Break" terminates a for without
        # executing the "else" clause.
        break
    else:
        # "Continue" starts the next iteration
        # of the loop. It's rather useless here,
        # as it's the last statement of the loop.
        continue

#do you understand this?

if rangelist[1] == 2:
    print("The second item (lists are 0-based) is 2")
else:
    pass

#list comprehensions ... powerfull shorthand stuff

erster = [1,2,3,4,5]
zweiter = [10, 100, 1000, 10000, 100000]
listComprehension = [x*y for x in erster for y in zweiter]
print(listComprehension)
len(listComprehension)

#here is a complicated example that I use very often
#generate a ternary
import itertools as it
n=10
inary=3
el = np.array([i/n for i in range(n+1)])
_comps = np.array([x for x in it.product(el, repeat=inary) if np.isclose(np.sum(x),1)])
_comps

#as a function this can be reused very easily
#you can sepify default values by setting them to a number or anything else
default = 3
def myFunc(x,y=default):
  z = x+y
  return z

#or even do complex stuff:
def genComp(n=20,inary=4):
    el = np.array([i/n for i in range(n+1)])
    _comps = np.array([x for x in it.product(el, repeat=inary) if np.isclose(np.sum(x),1)])

xy = genComp(n=10,inary=3)

#try to understand this line:
sum([1 for i in [6, 5, 4, 4, 9] if i == 4])

#bad programming
def crazyFunc(a, b, addOne=False, additor=0):
    #return some fraction of a/b+1+n
    if addOne==True:
        return a/b+1
    elif additor!=0:
        return a/b+additor
    else:
        return a/b

#better ... yet not good
def crazyFunc2(a, b, addOne=False, additor=0):
    #return some fraction of a/b+1+n
    if addOne==True:
        z = a/b+1
    elif additor!=0:
        z = a/b+additor
    else:
        z = a/b
    return z

z = crazyFunc(3,2,addOne=True,additor=1)
print(z)

#try running this code

def fehlerfehler():
    try:
        1 / 0
        #das universum kaputt
    except ZeroDivisionError:
        print("Nix da duch Null teilen.")
    else:
        pass
        #you may pass
    finally:
        #finally something is being done
        print("Noch was gemacht.")

fehlerfehler()

#objects are powerful things that can store values and functions and be initialized
#don't need to understand this for now 100%
class meineKlasse(object):
    allgeminErreichbar = 10
    def __init__(self):
        self.meineVariable = 3
    def meineFunktioninMeinerKlasse(self, arg1, arg2):
        return self.meineVariable

#There is one strangeness (and others) in python regarding scope ...
def ändertNix():
    # This will correctly change the global.
    x = 3

def ändert():
    global x
    # This will correctly change the global.
    x = 3

x = 2
ändertNix()
print(x)
ändert()
print(x)

#same thing like this:
a = 3
b = 2
c = a+b
print(c)
b=8
print(c)#whatt??

#finally ploting
import matplotlib.pyplot as plt
plt.plot([1,2,3,4], [1,4,9,16], 'ro')
plt.axis([0, 6, 0, 20])
plt.ylabel('Y LABEL')
plt.xlabel('xxx')
plt.show()

#ok now you ahave come this far and can run some really complicated stuff ... use it to play around a bit!

import numpy as np
import matplotlib.pyplot as plt

# Fixing random state for reproducibility
np.random.seed(1337)

mu, sigma = 100, 15
x = mu + sigma * np.random.randn(10000)

# the histogram of the data
n, bins, patches = plt.hist(x, 50, normed=1, facecolor='g', alpha=0.75)
plt.xlabel('Xlabel')
plt.ylabel('Ylabel')
plt.title('Title')
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.axis([40, 160, 0, 0.03])
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import NullFormatter  # useful for `logit` scale

# Fixing random state for reproducibility
np.random.seed(1337)

# make up some data in the interval ]0, 1[
y = np.random.normal(loc=0.5, scale=0.4, size=1000)
y = y[(y > 0) & (y < 1)]
y.sort()
x = np.arange(len(y))

plt.plot(x, y - y.mean())
plt.yscale('log', linthreshy=0.01)
plt.title('log')
plt.grid(True)
plt.show()
#fibonacchi example
#no goot list comprehension way found
fib = np.ones([20,1])
for i in range(2,20):
    fib[i] = fib[i-1] + fib[i-2]
print(fib)

#fibonacchi example
#caluclate the ratio
fib = np.ones([100,1])
ratio = np.empty([100,1])
ratio[0] = 1
ratio[1] = 1
for i in range(2,100):
    fib[i] = fib[i-1] + fib[i-2]
    ratio[i] = fib[i]/fib[i-1]
print(ratio)
from matplotlib import pyplot as pyplot

fig = plt.figure(figsize=[10,5])
ax = plt.subplot(111)
#ax = plt.gca()
i = [j for j in range(100)]
plot = plt.plot(i,ratio)
# use keyword args
plt.setp(plot, marker='o', color='k', linewidth=0.5)
ax.axis([1, 100, 0.9, 2.1])
ax.set_xscale("log", nonposx='clip')
#for axis in ['top','bottom','left','right']:
#  ax.spines[axis].set_linewidth(2.5)
plt.show()

#functional programming
#python is a multi paradigm language
#for developing algorithms it is sometimes useful
#to know what functional programming is
#here are some basic concepts

#iterators

h = iter(range(5))
print(h)
print(next(h))
print(next(h))
print(next(h))
print(next(h))

#generators - functions that create iterators i.e. resumable functions
def generate_squares(N):
    for i in range(N):
        yield i**2

sq = generate_squares(10)
print(next(sq))
print(next(sq))
print(next(sq))
print(next(sq))
print(next(sq))

#lambda - very shorthand one line functions

add = lambda x, y: x + y
multiply = lambda x,y : x*y
square = lambda x : x**2
isgreater = lambda x,y : x>y

print(add(3,2))
print(multiply(2.5,2))
print(isgreater(2.5,2))

#example
a = [(1, 2), (4, 1,3), (9, 10,6,7,8), (-1,13, -3)]
a.sort(key=lambda x: x[-1])

#map - apply a function to a list
_squared = map(square,[i for i in range(100)])
#squared = [s for s in _squared]

_cubed = map(lambda x: x**3, [i for i in range(100)])

def fsquare(x):
    return x**2
def fqube(x):
    return x**3

calcs = [fsquare,fqube]
_manycalcs = map(calcs, [i for i in range(10)])

#filters
even_nums = filter(lambda x: x % 2 == 0, range(30))
val = [k for k in even_nums]
val

#reduce - rolling excecution of functions on lists
#recommended to use for loops but conceptualy important
from functools import reduce
vecsum = reduce(lambda x, y : x+y, [1,-1,1,-1])

#example section

#gauss sum explicit formula summe = (n**2+n)/2
#with for loop
n = 100
summe = 0
for i in range(n+1):
    summe += i

#via list comprehension
n=100
summe = sum([i for i in range(n+1)])

#via reduce
n = 100
summe = reduce(lambda x, y : x+y, [i for i in range(n+1)])



#generate a multinary using generators
import itertools as it
import numpy as np
from functools import reduce

def steps(n):
    for i in range(n+1):
        yield i/n

def multinary_gen(n=10,inary=3):
    _el = steps(n)
    for x in it.product(_el, repeat=inary):
        if np.isclose(np.sum(x),1):
            yield x

comps_gen = multinary_gen(n=10,inary=3)

#calculate the number of compositions in an inary
comps_gen = multinary_gen(n=10,inary=3)
vecsum = reduce(lambda x, y : x+y, comps_gen)
print(len(vecsum)/3)

#just for fun
import multiprocessing
pool = multiprocessing.Pool()
def f(x):
    return x**2
print(pool.map(f, range(10)))
