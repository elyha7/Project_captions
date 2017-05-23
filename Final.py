"""@package docstring
Interface module
starts tkinter api
"""
import numpy as np
import pickle
from tqdm import tqdm
import sys
import numpy as np
from network import Network,load_data,splitter,make_vocabulary
from Tkinter import *
import tkFileDialog
from PIL import ImageTk, Image
from resizeimage import resizeimage
Net=0
root=Tk()
def main(argv):
	"""
	loads data, initializes network and starts api
	"""
	global Net
	img_codes,captions=load_data()
	captions=splitter(captions)
	print(captions.shape)
	captions=captions.flatten()
	#vocab,n_tokens,word_to_index=make_vocabulary(captions)
	vocab=pickle.load(open( "outfile1", "rb" ))
	#print(vocab[1])
	n_tokens=len(vocab)
	word_to_index=pickle.load(open( "outfile2", "rb" ))
	#print(word_to_index.keys()[0:5])
	PAD_ix = -1
	UNK_ix = vocab.index('#UNK#')
	Net=Network(img_codes,n_tokens,captions,vocab,word_to_index)
	Net.Network_init()
	Net.save_weights('model_big.npz',action='load')
	#Net.Network_train(5,100,10)
	#Net.make_caption('sample_images/stas.jpg')
	button1 = Button(root, font="helvetica 15", text="Open file")
	button1.grid(row=0, column=0)
	button1.bind("<Button-1>", Open)
	
	root.geometry("1000x480")
	root.mainloop()
def Open(event):
	"""
	action, that loads image, calls network function, that generates caption
	and shows image and caption in the box
	"""
	ftypes = [('JPEG files', '*.jpg'), ('All files', '*')]
	dlg = tkFileDialog.Open(filetypes = ftypes)
	fl = dlg.show()
        img_op = Image.open(fl)
        img_op = resizeimage.resize_contain(img_op, [400, 400])
        img = ImageTk.PhotoImage(img_op)
	outp = Net.make_caption(fl)
	label1 = Label(root, image = img)
	label1.image = img
        label2 = Label(root, font="helvetica 18")
        label3 = Label(root, font="helvetica 15")
        label4 = Label(root, font="helvetica 15")
        label5 = Label(root, font="helvetica 15")
        label2["text"] = "Captions:"
        label3["text"] = outp[0]
        label4["text"] = outp[1]
        label5["text"] = outp[2]
        label1.place(x=10, y=50)
        label2.place(x=450, y=150)
        label3.place(x=450, y=200)
        label4.place(x=450, y=250)
        label5.place(x=450, y=300)

if __name__ == '__main__':
	main(sys.argv)

