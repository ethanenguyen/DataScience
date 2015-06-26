# first we ingest the data from the source on the web
# this contains a reduced version of the data set from Lending Club
import pandas as pd
import sys
if "C:\\My_Python_Lib" not in sys.path:
    sys.path.append("C:\\My_Python_Lib")

import matplotlib.pyplot as plt
import pylab as pl
import numpy as np
import scipy
import sklearn

#from sklearn import datasets, linear_model
#import statsmodels.api as sm
import math



plt.figure()
loansData = pd.read_csv('C:\Users\uf425c\AppData\Local\Enthought\Canopy\loansData.csv')

loansData['Interest.Rate'][0:5] # first five rows of Interest.Rate

loansData['FICO.Range'][0:5]

loansData['Loan.Length'][0:5]

fico = loansData['Amount.Requested']
p = fico.hist()




s = "Hello world, world"
print len(s)

s2 = s.replace("world", "python")
s3 = s2.replace("Hello","monty")
print s2
print s3

l = range(-10,20,2)



