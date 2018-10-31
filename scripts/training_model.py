from pickle import load
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.vis_utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.merge import concatenate



#Load and clean dataset
def load_dataset(f_name):
	return load(open(f_name,'rb'))

#fit a tokenizer for the lines
def create_tokenizer(lines):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

#Calculate max length
def cal_max_length(lines):
	return max([len(s.split()) for s in lines ])

#Encode a list of lines
def encode_texts(tokenizer,lines,length):
	#integer encode or sequence encode
	encode_text = tokenizer.texts_to_sequences(lines)
	
	#Pad encode the sequence encode texts
	pad_encode_text = pad_sequences(encode_text,maxlen = length, padding = 'post')

	#return the encoded text
	return pad_encode_text



# Prepare 3 - layer CNN model
def define_CNN_model(length,vocab_size):
	#define channel 1 with 32 filter and 4-gram kernel size
	input_1 = Input(shape = (length,))
	embedding_1 = Embedding(vocab_size,100)(input_1)
	conv_1 = Conv1D(filters = 32,kernel_size = 4 , activation = 'relu')(embedding_1)
	drop_1 = Dropout(0.5)(conv_1)
	pool_1 = MaxPooling1D(pool_size = 2)(drop_1)
	flat_1 = Flatten()(pool_1)

	#define channel 2 with 32 filter and 6-gram kernel size
	input_2 = Input(shape = (length,))
	embedding_2 = Embedding(vocab_size,100)(input_2)
	conv_2 = Conv1D(filters = 32,kernel_size = 4 , activation = 'relu')(embedding_2)
	drop_2 = Dropout(0.5)(conv_2)
	pool_2 = MaxPooling1D(pool_size = 2)(drop_2)
	flat_2 = Flatten()(pool_2)
	#define channel 3 with 32 filter and 8-gram kernel size
	input_3 = Input(shape = (length,))
	embedding_3 = Embedding(vocab_size,100)(input_3)
	conv_3 = Conv1D(filters = 32,kernel_size = 4 , activation = 'relu')(embedding_3)
	drop_3 = Dropout(0.5)(conv_3)
	pool_3 = MaxPooling1D(pool_size = 2)(drop_3)
	flat_3 = Flatten()(pool_3)

	#merge the channel2
	merged = concatenate([flat_1,flat_2,flat_3])

	#Interpretation
	dense_1 = Dense(10,activation = 'relu')(merged)
	output = Dense(1,activation = 'sigmoid')(dense_1)
	model = Model(input = [input_1,input_2,input_3], outputs = output)
	print(model)
	#compile the model
	model.compile(loss = 'binary_crossentropy',
			optimizer = 'adam',
			metrics = ['accuracy'])

	#Summarize the model
	print(model.summary())
	#plot_model(model,show_shapes = False, to_file = 'multichannel.png')
	return model






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
