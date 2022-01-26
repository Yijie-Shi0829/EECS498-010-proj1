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
import nltk
from statsmodels.graphics.gofplots import qqplot


# 1.1 (2 pts.)
def findZeroDollarPrice(bmdf):
    return bmdf[bmdf["dollar_price"] == 0]

#  1.2 (1 pts.)


def replaceZeroWithNaN(bmdf):
    new = bmdf.copy()
    new["dollar_price"].replace(to_replace=0.0, value=np.NaN, inplace=True)
    return new

# 1.3 (1 pts.)


def replaceNaNWithZero(bmdfTest):
    new = bmdfTest.copy()
    new["dollar_price"].replace(to_replace=np.NaN, value=0.0, inplace=True)
    return new

# 1.4 (1 pts.)


def replaceNaNWithMean(bmdfTest):
    print(bmdfTest["dollar_price"].mean())
    new = bmdfTest.copy()
    new["dollar_price"].replace(
        to_replace=np.NaN, value=bmdfTest["dollar_price"].mean(), inplace=True)
    return new

# 1.5 (1 pts.)


def replaceNaNWithMedian(bmdfTest):
    new = bmdfTest.copy()
    new["dollar_price"].replace(
        to_replace=np.NaN, value=bmdfTest["dollar_price"].median(), inplace=True)
    return new

# 1.6 (2 pts.)


def replaceNaNWithNormal(bmdfTest):
    new = bmdfTest.copy()
    new["dollar_price"].replace(to_replace=np.NaN, value=np.random.normal(
        loc=bmdfTest["dollar_price"].mean(), scale=bmdfTest["dollar_price"].std()), inplace=True)
    return new

# 1.7 (3 pts.)


def graphIndepVsDep(bmdf, indepCols):
    for var in indepCols:
        plt.figure()
        plt.xlabel(var)
        plt.ylabel("dollar_price")
        plt.title(var+" vs dollar_price")
        plt.scatter(bmdf[var], bmdf["dollar_price"])
        plt.show()


# 1.8 (1 pts.)
def localPriceOutlier(bmdf):
    return bmdf[bmdf["local_price"] > 60000]

# 1.9 (1 pts.)


def dollarExOutliers(bmdf):
    return bmdf[bmdf["dollar_ex"] > 20000]

# 1.10 (3 pts.)


def correlationHeatmap(bmdf):
    attrs = []
    for attr in bmdf.columns.to_list():
        if pd.api.types.is_numeric_dtype(bmdf[attr]):
            attrs.append(attr)
    corr = bmdf[attrs].corr()
    plt.matshow(corr)

# 1.11 (3 pts.)


def linearRegressionFit(bmdf):
    model = ols(formula='dollar_price ~ GDP_dollar+USD', data=bmdf).fit()
    print(model.summary())
    return model

# 1.12 (3 pts.)


def graphIndepVsResidual(bmdf, indepCols, model):
    for var in indepCols:
        plt.figure()
        plt.xlabel(var)
        plt.ylabel("residual")
        plt.title(var+" vs residual")
        plt.scatter(bmdf[var], bmdf["residual"])
        plt.show()

# 1.13 (5 pts.)


def histOfResiduals(bmdf):
    _ = plt.hist(bmdf["residual"], bins='auto', density=True,
                 histtype='stepfilled', alpha=0.2)
    mu, sigma = norm.fit(bmdf["residual"])
    X = np.linspace(-1.5, 1.5, 100)
    Y = norm.pdf(X, mu, sigma)
    plt.plot(X, Y)
    plt.xlabel("residual")
    plt.ylabel("number")
    plt.title("Histogram of residual")
    plt.show()

# 1.14 (3 pts.)


def graphQQPlot(model):
    qqplot(model.resid, fit=True)
    X = []
    for a in np.linspace(-3, 3, 100):
        X.append(a)
    plt.plot(X, X)
    plt.show()

# 1.15 (3 pts.)


def graphFittedVsResidual(model):
    plt.scatter(model.fittedvalues, model.resid)
    plt.title("fitted value vs residual")
    plt.xlabel("fitted value")
    plt.ylabel("residual")
    plt.show()

# 1.16 (2 pts.)


def predictLinearRegression(model, gdp_dollar, usd):
    data = pd.DataFrame({"GDP_dollar": [gdp_dollar], "USD": [usd]})
    return model.predict(data)[0]


#############################################################################
############################ PART 2 #########################################
#############################################################################


# 2.1 (3 pts.)
def lowerCase(df):
    def lower(str):
        new_str = ""
        for i in str:
            if i.isupper():
                new_str += i.lower()
            else:
                new_str += i
        return new_str
    df.text = df.text.map(lower)
    return df

# 2.2 (3 pts.)


def tokenizeDF(df):
    df.text = df.text.map(nltk.word_tokenize)
    return df

# 2.3 (4 pts.)


def removeStop(df, stopeng):
    def remove(wordlist):
        newwordlist = []
        for word in wordlist:
            if word not in stopeng:
                newwordlist.append(word)
        return newwordlist
    df.text = df.text.map(remove)
    return df

# 2.4 (4 pts.)


def keepAlpha(df):
    def newlist(wordlist):
        newwordlist = []
        for word in wordlist:
            flag = False
            if len(word) > 1:
                for char in word:
                    if char.isalpha():
                        flag = True
                        break
                if (flag):
                    newwordlist.append(word)
                else:
                    continue
        return newwordlist
    df.text = df.text.map(newlist)
    return df


# 2.5 (3 pts.)
def joinText(df):
    def join(wordlist):
        return ' '.join(wordlist)
    df.text = df.text.map(join)
    return df

# 2.6 (4 pts.)


def countVectorize(df):
    cv = CountVectorizer()
    cv_fit = cv.fit_transform(df.text)
    X = cv_fit.toarray()
    y = np.array(df.label)
    vectorizer = cv
    return X, y, vectorizer

# 2.7 (3 pts.)


def splitData(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)
    return X_train, X_test, y_train, y_test

# 2.8 (4 pts.)


def trainLogisticRegression(X_train, X_test, y_train, y_test):
    clf = LogisticRegression(random_state=42).fit(X_train, y_train)
    return clf, clf.score(X_test, y_test)

# 2.9 (4 pts.)


def scoreText(clf, vectorizer, text):
    vec = vectorizer.transform([text])
    return clf.predict(vec)[0], clf.decision_function(vec)[0]

# 2.10 (3 pts.)


def tfidfVectorize(df):
    tv = TfidfVectorizer()
    tv_fit = tv.fit_transform(df.text)
    X = tv_fit.toarray()
    y = np.array(df.label)
    vectorizer = tv
    return X, y, vectorizer
