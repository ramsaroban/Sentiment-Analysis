import imp
from string import punctuation
from os import listdir
from nltk.corpus import stopwords
from pickle import dump

#Load Data into memory 
def load_data(f_name):
	#open file in file read mood
	file = open(f_name,'r')
	#read file and store it in text
	text = file.read()
	#close file (it is always good to close file after operations)
	file.close()
	#return text
	return text



#The second most important thing is to clean data and tokenize it.
# Cleaning of data means remove white space, .,a,an,the,stopwords,word length <= 1.
def clean_data(texts):
	#Splite the data into tokens by white space 
	tokens = texts.split()
	
	#remove puntuation from each tokens
	tab = str.maketrans('','',punctuation)
	tokens = [w.translate(tab) for w in tokens]

	#Remove tockens that are not alphabatics
	tokens = [w for w in tokens if w.isalpha()]
	
	#Filter out stop words like between,where, etc.
	stop_word = set(stopwords.words('english'))
	tokens = [w for w in tokens if not w in stop_word]
	
	#filter out short tokens
	tokens = [w for w in tokens if len(w) > 1]

	tokens = ' '.join(tokens)

	#return cleaned data as tokens
	return tokens
	

# Processing of the data, read all training data from the given path
def process_data(path,is_train):
	#A list type of documents which will store all the data after processing it.
	documents = list()
	sample = 0
	#Walk through all the files in the folder\
	for f_name in listdir(path):
		#Skip any reviews in the test set
		if is_train and f_name.startswith('cv9'):
			continue
		if not is_train and not f_name.startswith('cv9'):
			continue
		#Generate full path for the files
		f_path = path +'/'+f_name
		
		#load the data 
		l_data = load_data(f_path)
		
		#Clean the data
		c_data = clean_data(l_data)
		
		#add all data to the main list 
		documents.append(c_data)
		sample += 1;
	return documents,sample	 	

#Save data to a file
def save_data(dataset,file_name):
	#dump data to the file given as argument
	dump(dataset,open(file_name,'wb'))
	print("Saved: %s" %file_name)
	
#text = load_data('utils.txt')
#texts = clean_data(text)		 
#print(texts)
