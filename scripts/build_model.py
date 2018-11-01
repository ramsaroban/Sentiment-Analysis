from __init__ import *


#load Training dataset
train_Lines, train_Labeles = load_dataset('../data/train.pkl')

#create tokenizer
tokenizer = create_tokenizer(train_Lines)

#Caculate length
length = cal_max_length(train_Lines)

#calculate Vocab size
vocab_size = len(tokenizer.word_index)+1
print("Max Document length is : %d " %length)
print("Vocab_size : %d " %vocab_size)

#Encode the dataset
trainX = encode_texts(tokenizer,train_Lines,length)
print("---------------Shape of the training dataset----------- ")
print(trainX.shape)


#define model
model = define_CNN_model(length, vocab_size)
#fit the model
model.fit([trainX,trainX,trainX],array(train_Labeles), epochs = 10, batch_size = 16)
#Save the model
model.save('model.h5') 
