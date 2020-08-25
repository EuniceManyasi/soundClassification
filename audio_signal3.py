# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 08:33:59 2020

@author: Administrator
"""

import pathlib
from pathlib import Path 

import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import datasets, layers, models


SAMPLE_RATE=16000;
LABELS=[];
batch_size=10;
DATASET=[];
TRAIN_AUDIO=[];
TRAIN_LABELS=[];

TEST_AUDIO=[];
TEST_LABELS=[];

TRAIN_FILES=[];

bpath='speech_commands_v0.01/';
   
          
def parse_audio(filename):
    parts = tf.strings.split(filename, os.sep);
    label = parts[-2];

    audio_file = tf.io.read_file(filename);
    pcm, sample_rate = tf.audio.decode_wav(audio_file,desired_channels=1,desired_samples=16000);
 
    return pcm[:,0], label;
   
def compute_stft(file,frame_length_s,frame_step_s): 
    length= int(frame_length_s*SAMPLE_RATE);
    step=int(frame_step_s* SAMPLE_RATE);
    
    stft = tf.signal.stft(file, window_fn=tf.signal.hann_window, 
    frame_length=length, frame_step=step);
    
    magnitude_spectrogram = tf.abs(stft);           
    
    return magnitude_spectrogram;


def mel_spectrogram(magnitude_spectrogram):
    #Warp the linear scale, magnitude spectrograms into the mel scale
    sample_rate=16000;
     
    num_spectrogram_bins = magnitude_spectrogram.shape[-1];
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 125.0, 7500.0, 128;
    
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
      num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz,
      upper_edge_hertz);
            
    mel_spectrogram = tf.tensordot(
      magnitude_spectrogram, linear_to_mel_weight_matrix, 1);
            
    mel_spectrogram=tf.math.log(mel_spectrogram);
            
    return mel_spectrogram;

def process_labels(label):
    kv_init = tf.lookup.KeyValueTensorInitializer(
    [ 'down', 'eight', 'five', 'four', 'go', 'left', 'nine', 'no', 'off',
     'on', 'one', 'right', 'seven', 'six', 'stop', 
     'three', 'two', 'up', 'yes', 'zero'], range(1,20,1));

    ht = tf.lookup.StaticHashTable(kv_init, 0);
    
    return ht.lookup(label);

def process_audio(filename):
    audio,label=parse_audio(filename);
    processed_label=process_labels(label);
    stft= compute_stft(audio,0.08,0.04);
    melspec= mel_spectrogram(stft);

    return melspec,processed_label;

def load_file(basepath):
    basepath_root = pathlib.Path(basepath);
    list_ds = tf.data.Dataset.list_files(str(basepath_root/'*/*'));
    return list_ds;


def hashfile(filename):
    hashed=tf.strings.to_hash_bucket_fast(filename, 10);
    return filename, hashed;
    
def train_test_split(bpath,is_train):
    list_ds=load_file(bpath);
    hashed=list_ds.map(hashfile);
    
    if is_train:
        hashed= hashed.filter(lambda f,b:b!=1);
    else:
        hashed= hashed.filter(lambda f,b:b==1);
    return hashed;

def processed_data(filename):
    
    processed_dataset=filename.map(process_audio);
    
    return processed_dataset;
    
      
def cnn_model(melspec):
    
   conv1 = tf.layers.conv2d(
      inputs=melspec,
      filters=128,
      kernel_size=[8, 1],
      padding="same",
      input_shape=(24, 1, 128),
      activation=tf.nn.relu);
    
   conv2 = tf.layers.conv2d(
      inputs=conv1,
      filters=128,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)
      
      
   pool1 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 1]);
   flatten_pool1=tf.layers.Flatten(pool1);
   output=tf.layers.Dense(flatten_pool1,31,activation=tf.nn.softmax);
   return output;
   

def model_run():
    train_set_files=train_test_split(bpath,True);
    #for filename, hashed in train_set_files:
        #TRAIN_FILES.append(filename);
 
    train_data=train_set_files.map(processed_data)
        
    #train_set_process=processed_data(train_set_files);
    
    for mel,labels in train_data.take(5):
        print(mel);
        print("");
        print(labels);
    #test_set_files=train_test_set(bpath,False);
    
    #train_data=processed_data(train_set_files);
    
    #return train_data;


out=model_run();
print(out);