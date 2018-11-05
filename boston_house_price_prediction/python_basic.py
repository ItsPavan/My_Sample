#number of occurences of 'ab' in 'abcdefabgh'
'''
s1 = 'abcdefabcgh'
s2 = 'abc'

def find_occurence(s1,s2):
    if s2 in s1:
        return s1.count(s2)

print (find_occurence(s1,s2))

#a = [1,2,3,4,5,6,7,8],b = [3,5,6,8] print c = [1,2,4,7](elements in a but not in b) using list comprehension
a,b = [1,2,3,4,5,6,7,8],[3,5,6,8]
c = [i for i in a if i not in b]
print (c)

print (list(set(a)-set(b)))

a,b = [1,2,3,4,5],['a','b','c','d','e']
#c = dict(zip(a,b)) #using zip because len of both lists are same,if unequal only len of amaller list will be considered
c = {i:j for (i,j) in a,b}
print (c)
#>>> min(timeit.repeat(lambda: {k: v for k, v in zip(keys, values)}))
##0.7836067057214677
#>>> min(timeit.repeat(lambda: dict(zip(keys, values))))
#1.0321204089559615
#>>> min(timeit.repeat(lambda: {keys[i]: values[i] for i in range(len(keys))}))
#1.0714934510178864
#>>> min(timeit.repeat(lambda: dict([(k, v) for k, v in zip(keys, values)])))
#1.6110592018812895
#>>> min(timeit.repeat(lambda: dict((k, v) for k, v in zip(keys, values))))
#1.7361853648908436
'''
'''
a = [4,5,6]

def change ():
    #global a
    a = [1,2,3]
    print (a)

if __name__=='__main__':
    change()
    print ('********')
    print (a)


def ifint(func):
    def calc_wrapper(x):
        print("Before calling " + func.__name__)
        func(x)
        print("After calling " + func.__name__)
        c = [i for i in x if type(i) == int]
        print (c)
    return calc_wrapper

@ifint
def calc(a):
    print (a)

calc([1,2,3,4,5,'a','b','c'])
'''
'''
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the plusMinus function below.
def plusMinus(arr):
    p,n,z = [],[],[]
    for i in arr:
        if i < 0:
            n.append(i)
        elif i > 0:
            p.append(i)
        else:
            z.append(i)
    print (len(p)/len(arr),len(n)/len(arr),len(z)/len(arr))
            
    

plusMinus([-4, 3, -9, 0, 4, 1])

'''
from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from functools import reduce
print (PorterStemmer().stem('runs'))
#print ('Stem',SnowballStemmer('english').stem('running'))
print ('Lemma',WordNetLemmatizer().lemmatize('runs'))
#fib = [0,1,1,2,3,5,8,13,21,34,55]
#r = lambda x,y : x+y if x%2==0 else y
#result = reduce(r, fib)
#print (result)