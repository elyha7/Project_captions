import theano
import lasagne
import theano.tensor as T
from lasagne.layers import *
import numpy as np
from broadcast import BroadcastLayer,UnbroadcastLayer
from random import choice
from tqdm import tqdm
from pretrained_lenet import build_model,preprocess,MEAN_VALUES
from matplotlib import pyplot as plt
import pickle
class Network(object):
	"""docstring for ClassName"""
	def __init__(self, img_codes,ntokens,captions,vocab,word_to_index):
		self.vocab=vocab
		self.PAD_ix = -1
		self.captions=np.array(captions)
		self.n_tokens=ntokens
		self.CNN_FEATURE_SIZE = img_codes.shape[1]
		self.EMBED_SIZE = 512 #pls change me if u want
		self.LSTM_UNITS = 800 #pls change me if u want
		self.word_to_index=word_to_index
		self.UNK_ix = self.vocab.index('#UNK#')
		self.img_codes=img_codes
	def Network_init(self):
		self.sentences = T.imatrix()# [batch_size x time] of word ids
		self.image_vectors = T.matrix() # [batch size x unit] of CNN image features
		sentence_mask = T.neq(self.sentences, self.PAD_ix)
		l_words = InputLayer((None, None), self.sentences)
		l_mask = InputLayer((None, None), sentence_mask)
		l_word_embeddings = EmbeddingLayer(l_words,input_size=self.n_tokens, output_size=self.EMBED_SIZE)
		#converting 1000 image features from googlenet to LSTM_UNITS
		l_image_features = InputLayer((None, self.CNN_FEATURE_SIZE), self.image_vectors)
		l_image_features_small = DropoutLayer(l_image_features,0.3)
		l_image_features_small = DenseLayer(l_image_features_small,self.LSTM_UNITS)
		assert l_image_features_small.output_shape == (None, self.LSTM_UNITS)
		#Black magic. Taking image features as hidden state of lstm and word embedings as input, 
		#also taking a mask to calculate loss correctly.
		decoder = LSTMLayer(l_word_embeddings,
                    num_units=self.LSTM_UNITS,
                    cell_init=l_image_features_small,
                    mask_input=l_mask,
                    grad_clipping=10)
		broadcast_decoder_ticks = BroadcastLayer(decoder, (0, 1))
		print "broadcasted decoder shape = ",broadcast_decoder_ticks.output_shape

		predicted_probabilities_each_tick = DenseLayer(
		    broadcast_decoder_ticks,self.n_tokens, nonlinearity=lasagne.nonlinearities.softmax)

		#un-broadcast back into (batch,tick,probabilities)
		self.predicted_probabilities = UnbroadcastLayer(
		    predicted_probabilities_each_tick, broadcast_layer=broadcast_decoder_ticks)

		print "output shape = ", self.predicted_probabilities.output_shape

		#remove if you know what you're doing (e.g. 1d convolutions or fixed shape)
		assert self.predicted_probabilities.output_shape == (None, None, self.n_tokens)

		next_word_probas = lasagne.layers.get_output(self.predicted_probabilities)
		val_probas = lasagne.layers.get_output(self.predicted_probabilities,deterministic=True)

		reference_answers = self.sentences[:,1:]
		output_mask = sentence_mask[:,1:]

		#Symbolic loss function to train NN for
		loss = lasagne.objectives.categorical_crossentropy(
		    next_word_probas[:, :-1].reshape((-1, self.n_tokens)),
		    reference_answers.reshape((-1,))
		).reshape(reference_answers.shape)
		val_loss = lasagne.objectives.categorical_crossentropy(
		    val_probas[:, :-1].reshape((-1, self.n_tokens)),
		    reference_answers.reshape((-1,))
		).reshape(reference_answers.shape)
		
		#calculating loss and validation loss over non-PAD tokens
		loss = (loss*output_mask).sum()/output_mask.sum()
		val_loss = (val_loss*output_mask).sum()/output_mask.sum()
		
		#Network weights and the function, that updates tham after loss computation, using gradient descent method Adam
		weights = lasagne.layers.get_all_params(self.predicted_probabilities)
		update_w = lasagne.updates.adam(loss,weights)
		
		#A function that takes input sentence and image mask, outputs loss and updates weights
		self.train_step = theano.function([self.image_vectors,self.sentences],loss, updates=update_w)
		self.val_step   = theano.function([self.image_vectors,self.sentences],val_loss)

	def Network_train(self,epoch_n,batch_size,batch_pe):
		batch_size = batch_size #adjust me
		n_epochs   = epoch_n #adjust me
		n_batches_per_epoch = batch_pe #adjust me
		n_validation_batches = 5 #how many batches are used for validation after each epoch
		for epoch in range(n_epochs):
		    train_loss=0
		    for _ in tqdm(range(n_batches_per_epoch)):
		        train_loss += self.train_step(*self.generate_batch(batch_size))
		    train_loss /= n_batches_per_epoch
		    
		    val_loss=0
		    for _ in range(n_validation_batches):
		        val_loss += self.val_step(*self.generate_batch(batch_size))
		    val_loss /= n_validation_batches
		    if epoch %5==0:
		        print('\nEpoch: {}, train loss: {}, val loss: {}'.format(epoch, train_loss, val_loss))

		print("Finish")

	def make_caption(self,image_path):

		# build googlenet
		self.lenet = build_model()

		#load weights
		self.lenet_weights = pickle.load(open('data/blvc_googlenet.pkl'))['param values']
		set_all_param_values(self.lenet["prob"], self.lenet_weights)

		#compile get_features
		self.cnn_input_var = self.lenet['input'].input_var
		self.cnn_feature_layer = self.lenet['loss3/classifier']
		self.get_cnn_features = theano.function([self.cnn_input_var], lasagne.layers.get_output(self.cnn_feature_layer))
		
		self.last_word_probas_det = get_output(self.predicted_probabilities,deterministic=False)[:,-1]
		self.get_probs = theano.function([self.image_vectors,self.sentences], self.last_word_probas_det)

		to_ret=[]
		img=plt.imread(image_path)
		img = preprocess(img)
		for i in range(3):
			to_ret.append(' '.join(self.generate_caption(img,t=1.,sample=False,max_len=100)[1:-1]))
		print(to_ret)
		return to_ret
	def generate_caption(self,image,caption_prefix = ("START",),t=1,sample=True,max_len=100):
	    image_features = self.get_cnn_features(image)
	    caption = list(caption_prefix)
	    for j in range(max_len):
	        
	        next_word_probs = self.get_probs(image_features,self.as_matrix([caption]) ).ravel()
	        #apply temperature
	        next_word_probs = next_word_probs**t / np.sum(next_word_probs**t)

	        if sample:
	            next_word = np.random.choice(self.vocab,p=next_word_probs) 
	        else:
	            next_word = self.vocab[np.argmax(next_word_probs)]

	        caption.append(next_word)

	        if next_word=="#END#":
	            break
	            
	    return caption
	def as_matrix(self,sequences,max_len=None):
		max_len = max_len or max(map(len,sequences))

		matrix = np.zeros((len(sequences),max_len),dtype='int32')+self.PAD_ix
		for i,seq in enumerate(sequences):
		    row_ix = [self.word_to_index.get(word,self.UNK_ix) for word in seq[:max_len]]
		    matrix[i,:len(row_ix)] = row_ix
		return matrix
	def save_weights(self,filename,action=None):
		if action=='save':
		    np.savez(filename, *lasagne.layers.get_all_param_values(self.predicted_probabilities))
		if action=='load':
		    with np.load(filename) as f:
		        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
		    lasagne.layers.set_all_param_values(self.predicted_probabilities, param_values)
	def generate_batch(self,batch_size,max_caption_len=None):
	    #sample random numbers for image/caption indicies
	    random_image_ix = np.random.randint(0, len(self.img_codes), size=batch_size)
	    
	    #get images
	    batch_images = self.img_codes[random_image_ix]
	    
	    #5-7 captions for each image
	    captions_for_batch_images = self.captions[random_image_ix]
	    
	    #pick 1 from 5-7 captions for each image
	    batch_captions = map(choice, captions_for_batch_images)
	    
	    #convert to matrix
	    batch_captions_ix = self.as_matrix(batch_captions,max_len=max_caption_len)
	    
	    return batch_images, batch_captions_ix

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







