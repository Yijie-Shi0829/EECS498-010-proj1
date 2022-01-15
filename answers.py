import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from numpy import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from scipy.stats import norm
from statsmodels.graphics.gofplots import qqplot

# 1.1 (2 pts.) 
def findZeroDollarPrice(bmdf):
	pass

#  1.2 (1 pts.)
def replaceZeroWithNaN(bmdf):
	pass

# 1.3 (1 pts.) 
def replaceNaNWithZero(bmdfTest):
	pass

# 1.4 (1 pts.) 
def replaceNaNWithMean(bmdfTest):
	pass

# 1.5 (1 pts.) 
def replaceNaNWithMedian(bmdfTest):
	pass

# 1.6 (2 pts.) 
def replaceNaNWithNormal(bmdfTest):
	pass

# 1.7 (3 pts.) 
def graphIndepVsDep(bmdf, indepCols):
	pass
# 1.8 (1 pts.) 
def localPriceOutlier(bmdf):
	pass

# 1.9 (1 pts.) 
def dollarExOutliers(bmdf):
	pass

# 1.10 (3 pts.) 
def correlationHeatmap(bmdf):
	pass

# 1.11 (3 pts.) 
def linearRegressionFit(bmdf):
	pass
# 1.12 (3 pts.) 
def graphIndepVsResidual(bmdf, indepCols, model):
	pass

# 1.13 (5 pts.) 
def histOfResiduals(bmdf):
	pass

# 1.14 (3 pts.) 
def graphQQPlot(model):
	pass

# 1.15 (3 pts.) 
def graphFittedVsResidual(model):
	pass

# 1.16 (2 pts.) 
def predictLinearRegression(model, gdp_dollar, usd):
	pass

#############################################################################
############################ PART 2 #########################################
#############################################################################


# 2.1 (3 pts.) 
def lowerCase(df):
	pass

# 2.2 (3 pts.) 
def tokenizeDF(df):
	pass

# 2.3 (4 pts.) 
def removeStop(df, stopeng):
	pass

# 2.4 (4 pts.) 
def keepAlpha(df):
	pass

# 2.5 (3 pts.) 
def joinText(df):
	pass

# 2.6 (4 pts.)
def countVectorize(df):
	pass

# 2.7 (3 pts.) 
def splitData(X, y):
	pass

# 2.8 (4 pts.) 
def trainLogisticRegression(X_train, X_test, y_train, y_test):
	pass

# 2.9 (4 pts.)
def scoreText(clf, vectorizer, text):
    pass

# 2.10 (3 pts.) 
def tfidfVectorize(df):
	pass

