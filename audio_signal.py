# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 15:15:45 2020

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 15:56:45 2020

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
    ['bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'four',
     'go', 'happy', 'house', 'left', 'marvin', 'nine', 'no', 'off',
     'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 
     'three', 'tree', 'two', 'up', 'wow', 'yes', 'zero',
     '_background_noise_'], range(1,32,1));

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

      
def final(bpath):
    list_ds=load_file(bpath);
    final_dataset=list_ds.map(process_audio);
    
    return final_dataset;


def process_data(bpath):
    
    file=load_file(bpath);
    dataset_size= len(list(file));
    
    train_set_size=int(0.80 * dataset_size);
    #testing_set=audio_dataset.skip(train_set_size);
    
    audio_dataset= final(bpath);
    training_set=audio_dataset.take(train_set_size);
    batched_train_data= training_set.batch(10);
    
    testing_set=audio_dataset.skip(train_set_size);
    batched_test_data= testing_set.batch(10);
    
    
    
    for audio_specs, label in batched_train_data:
        TRAIN_AUDIO.append(audio_specs);
        TRAIN_LABELS.append(label);
        
    for audio_specs, label in batched_test_data:
        TEST_AUDIO.append(audio_specs);
        TEST_LABELS.append(label);
        
    return TRAIN_AUDIO,TRAIN_LABELS;
    
    #return training_set;
    
def process_model(melspecs):
    model = models.Sequential()
    hiddens= model.add(layers.Conv2D(melspecs,128, (8, 1), activation='relu', input_shape=(24, 1, 128),padding="same"))
    hiddens2= model.add(layers.Conv2D(hiddens,128, (3, 3), activation='relu',padding="same"))
    hiddens3=model.add(layers.MaxPooling2D((hiddens2,2, 1)))
    hidden4=model.add(layers.Flatten(hiddens3))
    output=model.add(layers.Dense(hidden4,31))
    
    
    train_audio,train_labels= process_data(bpath);
    #batched_train_data= training_set.batch(10);
    
    model.compile(optimizer='adam',
              loss="sparse_categorical_crossentropy",
              metrics=['accuracy'])

    history = model.fit(train_audio, train_labels, epochs=1, 
                    validation_data=(test_audio, test_labels))
      
    return(history);



pd=process_data(bpath);
print(pd);

    





 



