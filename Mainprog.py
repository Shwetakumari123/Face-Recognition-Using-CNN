import tkinter
from tkinter import *
import subprocess
import os
from os import system as cmd

WINDOW_SIZE = "800x700"
top = tkinter.Tk()


top.geometry(WINDOW_SIZE)
top.title("Face Recognition")



w = tkinter.Label(top, text="Face Recognition System",fg = "light green", bg = "dark green", font = "Helvetica 20 bold italic")
w.pack()


def Take_image():   
   os.system('xterm -into %d -geometry 150x30 -sb -e python3 camera2.py &' % wid)
   
def Train_image():   
   os.system('xterm -into %d -geometry 150x30 -sb -e python3 mycnnfinal_train.py &' % wid)
def Predict_image():   
   os.system('xterm -into %d -geometry 150x30 -sb -e python3 predict.py &' % wid)

Take_image1  = tkinter.Button(top, text =" Capture_Image_for_Retraining_the_Model ",fg = "light blue", bg = "dark blue", font = "Helvetica 16 bold italic", command = Take_image)
Take_image1.pack()

Train_image1  = tkinter.Button(top, text ="Retrain_the_model ",fg = "light blue", bg = "dark blue", font = "Helvetica 16 bold italic", command = Train_image)
Train_image1.pack()


Take_image1  = tkinter.Button(top, text =" Capture_Image_for_Prediction",fg = "light blue", bg = "dark blue", font = "Helvetica 16 bold italic", command = Take_image)
Take_image1.pack()



Predict_image1  = tkinter.Button(top, text ="Predict_Image",fg = "light blue", bg = "dark blue", font = "Helvetica 16 bold italic", command = Predict_image)
Predict_image1.pack()

termf = Frame(top, height=800, width=600)

termf.pack(fill=BOTH, expand=YES)
wid = termf.winfo_id()

os.system('xterm -into %d -geometry 220x30 -sb &' % wid)

def send_entry_to_terminal(*args):
    
    cmd("%s" % (BasicCovTests))

top.mainloop()
