from __init__ import *


####################################################### Train data prepration started################
print("Prepraring for training data started.........")
#Load all Training revies and store it in a training file for further use
neg_doc,neg_sample = process_data('../data/neg',True)
print("Total Negative reviews sample for training is : %d" %neg_sample)
pos_doc,pos_sample = process_data('../data/pos',True)
print("Total Positive reviews sample for training is : %d" %pos_sample)

#Data for X and Y co-ordinates
trainX = neg_doc + pos_doc
trainY = [0 for _ in range(neg_sample)] + [1 for _ in range(pos_sample)]
print("Saving training data.......")
save_data([trainX,trainY],'../data/train.pkl')
print("Training data Prepration done.........")
####################################################### Test data prepration started################
print("Prepraring for testing data started.........")
#Load all testing revies and store it in a testing file for further use
neg_doc,neg_sample = process_data('../data/neg',False)
print("Total Negative reviews sample for testing is : %d" %neg_sample)
pos_doc,pos_sample = process_data('../data/pos',False)
print("Total Positive reviews sample for testing is : %d" %pos_sample)

#Data for X and Y co-ordinates
trainX = neg_doc + pos_doc
trainY = [0 for _ in range(neg_sample)] + [1 for _ in range(pos_sample)]
print("Saving testing data.......")
save_data([trainX,trainY],'../data/test.pkl')
print("Testing data Prepration done.........")
