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
import cv2
import time
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
from IPython.display import SVG
from keras.utils.vis_utils import plot_model


model = load_model("../models/muzaffar.h5")
model.load_weights("music_tagger_cnn.h5")
def analize():

 #plot_model(model, to_file='model_plot1.png', show_shapes=False, show_layer_names=True)
 #SVG(model_to_dot(model).create(prog='dot', format='svg'))
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
clicks = 0
file_name = ""
file_name1 = ""
id = 0
stop = False


def play1():
  pygame.mixer.init()
  pygame.mixer.music.load(file_name1)
  musictime = pygame.mixer.music.get_pos()
  pygame.mixer.music.play()
  time.sleep(10)
  pygame.mixer.music.stop()


def main():

    global id
    #print(y)
    if(id == 0):
      print("metal")
      time.sleep(1)
      img = cv2.imread('metal.jpg')
      cv2.imshow('image',img)
      cv2.waitKey(0)
      #cv2.destroyAllWindows()
      #im.close()
    if(id==1):
      print("disco")
      time.sleep(1)
      img = cv2.imread('disco.jpg')
      cv2.imshow('image',img)
      cv2.waitKey(0)
      #cv2.destroyAllWindows()
    if(id==2):
      print("classical")
      time.sleep(1)
      img = cv2.imread('classical.jpg')
      cv2.imshow('image',img)
      cv2.waitKey(0)
      #cv2.destroyAllWindows()
    if(id == 3):
      print("hiphop")
      time.sleep(1)
      img = cv2.imread('hiphop.jpg')
      cv2.imshow('image',img)
      cv2.waitKey(0)
      #cv2.destroyAllWindows()
    if(id == 4):
      print("jazz")
      time.sleep(1)
      img = cv2.imread('jazz.jpg')
      cv2.imshow('image',img)
      cv2.waitKey(0)
      #cv2.destroyAllWindows()



def click_buttom1():
    global clicks
    global file_name
    global file_name1
    global id
    file_name = '1.au'
    file_name = "../dataset/123/" + file_name
    file_name1 = '1.mp3'
    id = analize()
    p = multiprocessing.Pool()

    p.map(lambda f: f(),[main, play1])
    p.close()
    p.join()


def click_buttom2():
    global clicks
    global file_name
    global file_name1
    global id
    file_name = '2.au'
    file_name = "../dataset/123/" + file_name
    file_name1 = '2.mp3'
    id = analize()
    p = multiprocessing.Pool()

    p.map(lambda f: f(),[main, play1])
    p.close()
    p.join()

def click_buttom3():
    global clicks
    global file_name
    global file_name1
    global id
    file_name = '3.au'
    file_name1 = '3.mp3'
    file_name = "../dataset/123/" + file_name
    id = analize()
    p = multiprocessing.Pool()

    p.map(lambda f: f(),[main, play1])
    p.close()
    p.join()

def click_buttom4():
    global clicks
    global file_name
    global file_name1
    global id
    file_name = '4.au'
    file_name1 = '4.mp3'
    file_name = "../dataset/123/" + file_name
    id = analize()
    p = multiprocessing.Pool()

    p.map(lambda f: f(),[main, play1])
    p.close()
    p.join()

def click_buttom5():
    global clicks
    global file_name
    global file_name1
    global id
    file_name = '5.au'
    file_name1 = '5.mp3'
    file_name = "../dataset/123/" + file_name
    id = analize()
    p = multiprocessing.Pool()

    p.map(lambda f: f(),[main, play1])
    p.close()
    p.join()

def click_buttom6():
    global clicks
    global file_name
    global file_name1
    global id
    file_name = '6.au'
    file_name1 = '6.mp3'
    file_name = "../dataset/123/" + file_name
    id = analize()
    p = multiprocessing.Pool()

    p.map(lambda f: f(),[main, play1])
    p.close()
    p.join()

def click_buttom7():
    global clicks
    global file_name
    global file_name1
    global id
    file_name = '7.au'
    file_name1 = '7.mp3'
    file_name = "../dataset/123/" + file_name
    id = analize()
    p = multiprocessing.Pool()

    p.map(lambda f: f(),[main, play1])
    p.close()
    p.join()

def click_buttom8():
    global clicks
    global file_name
    global file_name1
    global id
    file_name = '8.au'
    file_name1 = '8.mp3'
    file_name = "../dataset/123/" + file_name
    id = analize()
    p = multiprocessing.Pool()

    p.map(lambda f: f(),[main, play1])
    p.close()
    p.join()

def click_buttom9():
    global clicks
    global file_name
    global file_name1
    global id
    file_name = '9.au'
    file_name1 = '9.mp3'
    file_name = "../dataset/123/" + file_name
    id = analize()
    p = multiprocessing.Pool()

    p.map(lambda f: f(),[main, play1])
    p.close()
    p.join()

def click_buttom10():
    global clicks
    global file_name
    global file_name1
    global id
    file_name = '10.au'
    file_name1 = '10.mp3'
    file_name = "../dataset/123/" + file_name
    id = analize()
    p = multiprocessing.Pool()

    p.map(lambda f: f(),[main, play1])
    p.close()
    p.join()

def click_buttom11():
    global clicks
    global file_name
    global file_name1
    global id
    file_name = '11.au'
    file_name1 = '11.mp3'
    file_name = "../dataset/123/" + file_name
    id = analize()
    p = multiprocessing.Pool()

    p.map(lambda f: f(),[main, play1])
    p.close()
    p.join()

def click_buttom12():
    global clicks
    global file_name
    global file_name1
    global id
    file_name = '12.au'
    file_name1 = '13.mp3'
    file_name = "../dataset/123/" + file_name
    id = analize()
    p = multiprocessing.Pool()

    p.map(lambda f: f(),[main, play1])
    p.close()
    p.join()

def click_buttom13():
    global clicks
    global file_name
    global file_name1
    global id
    file_name = '13.au'
    file_name1 = '13.mp3'
    file_name = "../dataset/123/" + file_name
    id = analize()
    p = multiprocessing.Pool()

    p.map(lambda f: f(),[main, play1])
    p.close()
    p.join()

def click_buttom14():
    global clicks
    global file_name
    global file_name1
    global id
    file_name = '14.au'
    file_name1 = '14.mp3'
    file_name = "../dataset/123/" + file_name
    id = analize()
    p = multiprocessing.Pool()

    p.map(lambda f: f(),[main, play1])
    p.close()
    p.join()

def click_buttom15():
    global clicks
    global file_name
    global file_name1
    global id
    file_name = '15.au'
    file_name1 = '15.mp3'
    file_name = "../dataset/123/" + file_name
    id = analize()
    p = multiprocessing.Pool()

    p.map(lambda f: f(),[main, play1])
    p.close()
    p.join()





root = Tk()
root.title("Музыкальный классификатор")
root.geometry("120x700")
btn1 = Button(text="1", background="#555", foreground="#ccc", padx="60", pady="8", font="20", activebackground="#666", activeforeground = "#ddd", command=click_buttom1)
btn1.pack()
btn2 = Button(text="2", background="#555", foreground="#ccc", padx="60", pady="8", font="16", activebackground="#666", activeforeground = "#ddd", command=click_buttom2)
btn2.pack()
btn3 = Button(text="3", background="#555", foreground="#ccc", padx="60", pady="8", font="16", activebackground="#666", activeforeground = "#ddd", command=click_buttom3)
btn3.pack()
btn4 = Button(text="4", background="#555", foreground="#ccc", padx="60", pady="8", font="16", activebackground="#666", activeforeground = "#ddd", command=click_buttom4)
btn4.pack()
btn5 = Button(text="5", background="#555", foreground="#ccc", padx="60", pady="8", font="16", activebackground="#666", activeforeground = "#ddd", command=click_buttom5)
btn5.pack()
btn6 = Button(text="6", background="#555", foreground="#ccc", padx="60", pady="8", font="16", activebackground="#666", activeforeground = "#ddd", command=click_buttom6)
btn6.pack()
btn7 = Button(text="7", background="#555", foreground="#ccc", padx="60", pady="8", font="16", activebackground="#666", activeforeground = "#ddd", command=click_buttom7)
btn7.pack()
btn8 = Button(text="8", background="#555", foreground="#ccc", padx="60", pady="8", font="16", activebackground="#666", activeforeground = "#ddd", command=click_buttom8)
btn8.pack()
btn9 = Button(text="9", background="#555", foreground="#ccc", padx="60", pady="8", font="16", activebackground="#666", activeforeground = "#ddd", command=click_buttom9)
btn9.pack()
btn10 = Button(text="10", background="#555", foreground="#ccc", padx="56", pady="8", font="16", activebackground="#666", activeforeground = "#ddd", command=click_buttom10)
btn10.pack()
btn11 = Button(text="11", background="#555", foreground="#ccc", padx="56", pady="8", font="16", activebackground="#666", activeforeground = "#ddd", command=click_buttom11)
btn11.pack()
btn12 = Button(text="12", background="#555", foreground="#ccc", padx="56", pady="8", font="16", activebackground="#666", activeforeground = "#ddd", command=click_buttom12)
btn12.pack()
btn13 = Button(text="13", background="#555", foreground="#ccc", padx="56", pady="8", font="16", activebackground="#666", activeforeground = "#ddd", command=click_buttom13)
btn13.pack()
btn14 = Button(text="14", background="#555", foreground="#ccc", padx="56", pady="8", font="16", activebackground="#666", activeforeground = "#ddd", command=click_buttom14)
btn14.pack()
btn15 = Button(text="15", background="#555", foreground="#ccc", padx="56", pady="8", font="16", activebackground="#666", activeforeground = "#ddd", command=click_buttom15)
btn15.pack()

root.mainloop()





#if __name__ == '__main__':
