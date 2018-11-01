from __init__ import *
import sys
import argparse

test = 'test'
train = 'train'
def main():
    parser = argparse.ArgumentParser(description="Evaluate Sentiment Analysis Model...")
    parser.add_argument(
            "--input-data",
            default="../data/train.pkl",
            help="Default: Evaluating using Training dataset..",
            )

    args = parser.parse_args()
    return args.input_data

def evaluate_train_test_model(chs,mode):
    #Load model
    print("Loading Model***********************")
    model = load_model('model.h5')
    print("Loading model done *****************")
    #load Training dataset
    train_Lines, train_Labeles = load_dataset("../data/train.pkl")

    #create tokenizer
    tokenizer = create_tokenizer(train_Lines)

    #Caculate length
    length = cal_max_length(train_Lines)

    #calculate Vocab size
    vocab_size = len(tokenizer.word_index)+1
    print("Max Document length is : %d " %length)
    print("Vocab_size : %d " %vocab_size)

    if mode is 1:
        #Encode the dataset
        trainX = encode_texts(tokenizer,train_Lines,length)
        print("---------------Shape of the training dataset----------- ")
        print(trainX.shape)
        #evaluate the model by training dataset itself
        loss,accu = model.evaluate([trainX,trainX,trainX], array(train_Labeles), verbose = 0)
        print("Loss : %d" %(loss*100))
        print("Accuarcy : %d" %(accu*100))

    if mode is 2:
        #Load dataset
        test_Lines, test_Labeles = load_dataset(chs)
        #Encode dataset
        testX = encode_texts(tokenizer,test_Lines,length)
        print("---------------Shape of the testing dataset----------- ")
        print(testX.shape)
        #evaluate the model by training dataset itself
        loss,accu = model.evaluate([testX,testX,testX], array(test_Labeles), verbose = 0)
        print("Loss : %d" %(loss*100))
        print("Accuarcy : %d" %(accu*100))



if __name__ == "__main__":
   chs = main()
   mode = 0

   if train in chs:
       mode = 1
       print("Evaluating model using trainig dataset (%s) ...." %chs)
   if test in chs:
       mode = 2
       print("Evaluating model using testing dataset (%s) ...." %chs)
   #Call evaluate function    
   evaluate_train_test_model(chs,mode)
