import numpy as np
import pickle
from tqdm import tqdm
import sys
import numpy as np
from network import Network
def load_data():
	img_codes = np.load("data/image_codes.npy")
	captions = pickle.load(open('data/caption_tokens.pcl', 'rb'))
	return img_codes,captions
def splitter(captions):
	for img_i in range(len(captions)):
	    for caption_i in range(len(captions[img_i])):
	        sentence = captions[img_i][caption_i] 
	        captions[img_i][caption_i] = ["#START#"]+sentence.split(' ')+["#END#"]
	captions=np.array(captions)
	return captions
def make_vocabulary(captions):
	word_counts={}
	for i in tqdm(captions):
	    for j in i:
	        for k in j:
	                try: word_counts[k]+=1
	                except: word_counts[k]=1
	vocab  = ['#UNK#', '#START#', '#END#']
	vocab += [k for k, v in word_counts.items() if v >= 5]
	n_tokens = len(vocab)
	assert 10000 <= n_tokens <= 10500

	word_to_index = {w: i for i, w in enumerate(vocab)}
	return vocab,n_tokens, word_to_index

def main(argv):
	img_codes,captions=load_data()
	captions=splitter(captions)
	print(captions.shape)
	captions=captions.flatten()
	vocab,n_tokens,word_to_index=make_vocabulary(captions)
	PAD_ix = -1
	UNK_ix = vocab.index('#UNK#')
	Net=Network(img_codes,n_tokens,captions,vocab,word_to_index)
	Net.Network_init()
	Net.save_weights('model_big.npz',action=None)
	Net.Network_train(5,100,10)
	Net.make_caption('sample_images/stas.jpg')



if __name__ == '__main__':
    main(sys.argv)

