import numpy as np
import pickle
from tqdm import tqdm
import sys
import numpy as np
from network import Network,load_data,splitter,make_vocabulary
from Tkinter import *
import tkFileDialog
from PIL import ImageTk, Image
Net=0
root=Tk()
def main(argv):
	global Net
	img_codes,captions=load_data()
	captions=splitter(captions)
	print(captions.shape)
	captions=captions.flatten()
	vocab,n_tokens,word_to_index=make_vocabulary(captions)
	PAD_ix = -1
	UNK_ix = vocab.index('#UNK#')
	Net=Network(img_codes,n_tokens,captions,vocab,word_to_index)
	Net.Network_init()
	Net.save_weights('model_big.npz',action='load')
	#Net.Network_train(5,100,10)
	#Net.make_caption('sample_images/stas.jpg')
	button1 = Button(root, font=40, text="Open file")
	button1.grid(row=0, column=0)
	button1.bind("<Button-1>", Open)


	root.geometry("1000x600")

	root.mainloop()
def Open(event):
		
	ftypes = [('JPEG files', '*.jpg'), ('All files', '*')]
	dlg = tkFileDialog.Open(filetypes = ftypes)
	fl = dlg.show()
	img = ImageTk.PhotoImage(Image.open(fl))
	outp = Net.make_caption(fl)
	label1 = Label(root, image = img)
	label1.image = img
	label1.grid(row=1, column=0)
	label2 = Label(root, font=30)
	label3 = Label(root, font=30)
	label2.grid(row=0, column=1)
	label3.grid(row=1, column=1)
	label2["text"] = "Captions:"
	label3["text"] = outp[0]


if __name__ == '__main__':
	main(sys.argv)

