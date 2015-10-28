# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 12:25:49 2015

@author: f400752
"""

import os
import pandas as pd
os.chdir('C:\Users\F400752\Desktop\GA Data Science\Yelp')

'''-------------------------------------------------------------------------'''
                                '''TASK 1'''
                            '''Open yelp.csv'''
'''-------------------------------------------------------------------------'''
yelpcsv = pd.read_table('yelp.csv', sep=',')                                    
yelpcsv = pd.DataFrame(yelpcsv)
yelpcsv.head()

 
'''-------------------------------------------------------------------------'''
                                '''TASK 1(b)'''
                        '''Open yelp_json.json*'''
'''*for reasons I'll discuss with you, I was unable to get the JSON version'''
'''-------------------------------------------------------------------------'''


yelpjson = open('yelp_json.txt').read().split('\n')

jsonlist = [review.replace('{',', "').replace(', "', '|').split('|') for review in yelpjson]
jsonlist = [jlist[2:12] for jlist in jsonlist]
jsonlist = [[item.split(':') for item in jlist] for jlist in jsonlist]
jsonlist = [[[str(istr).strip('"').strip(' "').strip('\\n').strip('}') for istr in item] for item in jlist] for jlist in jsonlist]
json_headers = [item[0] for item in jsonlist[0]]
jsonlist = 

'''Deal with missing values'''

missings_busid = []
for jlist in jsonlist:
    for item in jlist:
        if item[0]=='business_id':
            missings_busid.append(jlist[3]) 
missings_type = []
for jlist in jsonlist:
    for item in jlist:
        if item[0]=='type':
            missings_type.append(jlist[3]) 

for jlist in jsonlist:
    for bID in missings_busid:  
        if jlist[3] <> bID:
            jlist.append(['business_id','NA'])
            
for jlist in jsonlist:
    for bID in missings_type:  
        if jlist[3] <> bID:
            jlist.append(['type','NA'])            
            
            
            
json_cols = ['review_id','funny','useful', 'cool', 'user_id', 'stars','date','text','type','business_id'] 



jsonlist_lists[2]

for attr in json_cols:
    attr = [attr]
    listoflists.append(attr)
    
for llist in listoflists:
    for jlist in jsonlist:
        for item in jlist:
            if item[0] == llist[0]:
                llist.append(item[1]) 
   
jsondict = {llist.pop(0):llist for llist in listoflists}

for key in jsondict.keys():
    print(key, len(jsondict[key]))

jsondict['business_id'].fillna(value='NA')
jsondict['type']


yelpDF = pd.DataFrame(jsondict)


'''-------------------------------------------------------------------------'''
                                '''TASK 2'''
'''Explore the relationship between each of the vote types (cool/useful/funny) 
                            and the number of stars.'''
# treat stars as a categorical variable and look for differences between groups
# correlation matrix as plt
'''-------------------------------------------------------------------------'''


%matplotlib inline
import matplotlib.pyplot
from sklearn.linear_model import LinearRegression

star_dummies = pd.get_dummies(yelpcsv.stars, prefix = 'star')                   
star_dummies.drop(star_dummies.columns[0], axis=1, inplace=True)

yelp = pd.concat([yelpcsv, star_dummies], axis =1)
type(yelp)
yelp.head(5)

feature_cols = [i for i in star_dummies]


'''"cool" votes'''
x = yelp[feature_cols]
y = yelp.cool
linreg = LinearRegression()
linreg.fit(x,y)
zip(feature_cols, linreg.coef_)


'''"funny" votes'''
x = yelp[feature_cols]
y = yelp.funny
linreg = LinearRegression()
linreg.fit(x,y)
zip(feature_cols, linreg.coef_)


'''"useful" votes'''
x = yelp[feature_cols]
y = yelp.useful
linreg = LinearRegression()
linreg.fit(x,y)
zip(feature_cols, linreg.coef_)


'''-------------------------------------------------------------------------'''
                                '''TASK 3'''
    '''Define cool/useful/funny as the features, and stars as the response.'''
'''-------------------------------------------------------------------------'''

feature_cols = ['cool','useful','funny']
x = yelp[feature_cols]
y = yelp.stars



'''-------------------------------------------------------------------------'''
                                '''TASK 4'''
        '''Fit a linear regression model and interpret the coefficients. 
    Do the coefficients make intuitive sense to you? Explore the Yelp website 
                    to see if you detect similar trends.'''
'''-------------------------------------------------------------------------'''

linreg = LinearRegression()
linreg.fit(x,y)
zip(feature_cols, linreg.coef_)

print linreg.intercept_



'''-------------------------------------------------------------------------'''
                                '''TASK 5'''
    '''Evaluate the model by splitting it into training and testing sets and 
        computing the RMSE. Does the RMSE make intuitive sense to you?'''
  # define a function that accepts a list of features and returns testing RMSE
                    # calculate RMSE with all three features
'''-------------------------------------------------------------------------'''

import numpy as np
from sklearn import metrics
yelp_shuffle = yelp.reindex(np.random.permutation(yelp.index))
yelp_shuffle.head()

def rmse(features):
    X = yelp_shuffle[features]
    y = yelp_shuffle.stars
    linreg = LinearRegression()
    linreg.fit(X,y)
    y_pred = linreg.predict(X)
    return np.sqrt(metrics.mean_squared_error(y, y_pred))

print "RMSE: ", rmse(feature_cols)



'''-------------------------------------------------------------------------'''
                                '''TASK 6'''
    '''Try removing some of the features and see if the RMSE improves.'''

'''-------------------------------------------------------------------------'''
print "cool and funny RMSE: ", rmse(feature_cols[::2])
print "cool and useful RMSE: ", rmse(feature_cols[0:1])
print "funny and useful RMSE: ", rmse(feature_cols[1:2])feature_cols[1])




'''-------------------------------------------------------------------------'''
                                '''TASK 7'''
    '''Think of some new features you could create from the existing 
    data that might be predictive of the response. Figure out how to 
    create those features in Pandas, add them to your model, and see 
                            if the RMSE improves.'''
# new feature: review length (number of characters)
# new features: whether or not the review contains 'love' or 'hate'
# add new features to the model and calculate RMSE
'''-------------------------------------------------------------------------'''
'''Review Length'''
review_length = pd.DataFrame([len(review) for review in yelp.text], columns=['review_length'])
love_review = pd.DataFrame([1 if "love" in review  else 0 for review in yelp.text], columns=['love_review'])
hate_review = pd.DataFrame([1 if "hate" in review  else 0 for review in yelp.text], columns=['hate_review'])
yelp = pd.concat([yelp, review_length, love_review, hate_review], axis=1)

yelp.head()

nfeature_cols = ['review_length', 'love_review', 'hate_review']
yelp_shuffle = yelp.reindex(np.random.permutation(yelp.index))

print "All RMSE: ", rmse(nfeature_cols)
print "review and love RMSE: ", rmse(nfeature_cols[0:1])
print "review and hate RMSE: ", rmse(nfeature_cols[::2])
print "love and hate RMSE: ", rmse(nfeature_cols[1:2])


'''-------------------------------------------------------------------------'''
                                '''TASK 8'''
    '''Compare your best RMSE on the testing set with the RMSE for the 
    "null model", which is the model that ignores all features and simply 
    predicts the mean response value in the testing set.'''
    
    # split the data (outside of the function)
    # create a NumPy array with the same shape as y_test
    # fill the array with the mean of y_test
    # calculate null RMSE
'''-------------------------------------------------------------------------'''
yelp_shuffle = yelp.reindex(np.random.permutation(yelp.index))

testing = yelp_shuffle[0:(len(yelp_shuffle)/2)]
training = yelp_shuffle[(len(yelp_shuffle)/2):]
len(testing)
len(training)
print testing


'''-------------------------------------------------------------------------'''
                                '''TASK 9'''
    '''Instead of treating this as a regression problem, treat it as a 
    classification problem and see what testing accuracy you can achieve 
    with KNN.'''
    
    # import and instantiate KNN
    # classification models will automatically treat the response value 
    #(1/2/3/4/5) as unordered categories
'''-------------------------------------------------------------------------'''




'''-------------------------------------------------------------------------'''
                                '''TASK 10'''
    '''Figure out how to use linear regression for classification, and 
        compare its classification accuracy with KNN's accuracy.'''
    
    # use linear regression to make continuous predictions
    # round its predictions to the nearest integer
    # calculate classification accuracy of the rounded predictions

'''-------------------------------------------------------------------------'''


















