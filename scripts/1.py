import gc
import os
import ast
import sys
import configparser
import librosa
import graphviz
import numpy as np
import pygame
from tkinter import *
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import keras
from keras import backend as K
from keras.models import load_model
from keras.utils import plot_model
from keras.utils.vis_utils import model_to_dot
from audiomanip.audiostruct import AudioStruct
from audiomanip.audiomodels import ModelZoo
from audiomanip.audioutils import AudioUtils
from audiomanip.audioutils import MusicDataGenerator
import multiprocessing.dummy as multiprocessing

def analize():
 model = load_model("../models/gtzan_hguimaraes.h5")
 model.load_weights("music_tagger_cnn.h5")
 song_samples = 660000

 sn_fft = 2048
 shop_length = 512
 sgenres = {'metal': 0, 'disco': 1, 'classical': 2, 'hiphop': 3, 'jazz': 4}
 signal, sr = librosa.load(file_name)

 # Calculate the melspectrogram of the audio and use log scale
 melspec = librosa.feature.melspectrogram(signal[:song_samples], sr = sr, n_fft = sn_fft, hop_length = shop_length).T[:128,]
 melspec = melspec[None,:]
 y = model.predict(melspec)
 pr = np.array(y)
 index = np.argmax(pr)
 return index
#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

clicks = 0
file_name = ""
file_name1 = ""
id = 0





def main():
  global id

  if(id == 0):
    print("metal")
    im = Image.open('metal.jpg')
    im.show()
  #im.close()
  if(id==1):
    print("disco")
    im = Image.open('disco.jpg')
    im.show()
    #im.close()
  if(id==2):
    print("classical")
    im = Image.open('classical.jpg')
    im.show()
    #im.close()
  if(id == 3):
    print("hiphop")
    im = Image.open('hiphop.jpg')
    im.show()
    #im.close()
  if(id == 4):
    print("jazz")
    im = Image.open('jazz.jpg')
    im.show()








def click_buttom1():
    global file_name1
    global file_name
    global id
    file_name = '../dataset/123/1.au'
    file_name1 = '1.mp3'
    id = analize()
    p = multiprocessing.Pool()

    p.map(lambda f: f(),[main, play1])
    p.close()
    p.join()

root = Tk()
root.title("Музыкальный классификатор")
root.geometry("300x400")
btn1 = Button(text="1", background="#555", foreground="#ccc", padx="60", pady="8", font="20", activebackground="#666", activeforeground = "#ddd", command=click_buttom1)
btn1.pack()
root.mainloop()




