from __init__ import *

print("Preparing dat.........")
pre_doc,pre_sample = process_data('../data/predict', False)
print("Sample: %d" %pre_sample)
preY = [0 for _ in range(pre_sample)] + [1 for _ in range(pre_sample)]
#preY = [0]+[1]
save_data([pre_doc,preY],"../data/predict.pkl")
print("Data prepration done........")

print("Prediction start..........")
#load test dataset
pre_Lines, pre_Labeles = load_dataset("../data/predict.pkl")
#print(pre_Lines)
#load Training dataset
train_Lines, train_Labeles = load_dataset("../data/train.pkl")

#create tokenizer
tokenizer = create_tokenizer(train_Lines)

#Caculate length
length = cal_max_length(train_Lines)
print("----Trainig data set information----")
#calculate Vocab size
vocab_size = len(tokenizer.word_index)+1
print("Max Document length is : %d " %length)
print("Vocab_size : %d " %vocab_size)
#Encode text
preX = encode_texts(tokenizer,pre_Lines,length)
#Load model
print("Loading Model***********************")
model = load_model('model.h5')
print("Loading model done *****************")
pre = model.predict([preX,preX,preX], verbose=0)
res =  round(pre[0,0])
if res == 1:
	print("Positive Review: [%d]" %res)
if res == 0:
	print("Negative Reviews: [%d]" %res)
