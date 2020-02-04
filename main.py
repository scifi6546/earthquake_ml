import tensorflow as tf
from keras.utils import to_categorical
import random

import numpy as np
import obspy as ob
import matplotlib
import matplotlib.pyplot as plt
from os import listdir
import json
def pick_emptytime(events,time_window,station):
    events_time = []
    for i in events:
        if station==i["station"]:
            events_time.append(i["time"])
    events_time.sort()
    empty_times = []
    if len(events_time)<=1:
        return []

    for i in range(0,len(events_time)-1):
        if events_time[i+1]-events_time[i]>time_window:
            empty_times.append((events_time[i+1]+events_time[i])/2)
    return empty_times


arrival_data=[]
num_samples = 15000;




with open("../../earthquake data/AEC/events.json") as f:
    arrival_data=json.load(f)
time_window =3.0

earthquake_folder = "../../earthquake data/AEC/2019/032"
files = listdir(earthquake_folder)
num_files = len(files);
i=0
wf_train=[]#np.ndarray(shape=(61))
phase_train=[]#np.ndarray(shape=(1),dtype=np.int32)
for eq_file in files:
    if(eq_file!="IRISDMC"):
        data = ob.read(earthquake_folder+"/"+eq_file)
        data.interpolate(60)
        
        if(i>=num_samples):
            break
        for event in arrival_data:
            if event["station"]==data[0].stats["station"]:
                rand_num = random.uniform(0,time_window/1.3);
                time_start = event["time"]-time_window+rand_num
                time_end = event["time"]+time_window+rand_num
                arrival = data.slice(ob.UTCDateTime(event["time"]-time_window)
                        ,ob.UTCDateTime(event["time"]+time_window))
                phase_array=np.array([0,0])
                normalized_time = (rand_num)/(2*time_window)+0.5
                    
                if(event["phase"]=='S'):
                    phase_array=np.array([1,normalized_time])
                if(event["phase"]=='P'):
                    phase_array=np.array([2,normalized_time])

                try:
                    print(arrival[0].data.shape)
                except:
                    print("index out of range")
                    break
                if arrival[0].data.shape==(361,):
                    if wf_train==[]:
                        wf_train=np.array(arrival[0].data)
                    else:
                        wf_train=np.vstack([wf_train,arrival[0].data])
                    print(wf_train.shape)
                    if(phase_train==[]):
                        phase_train=phase_array
                        print("phase_train empty")
                    else:
                        phase_train=np.vstack([phase_train,phase_array])
                break;
        empty_times = pick_emptytime(arrival_data,time_window*2,data[0].stats["station"])
        for time in empty_times:
            arrival = data.slice(ob.UTCDateTime(time-time_window)
                ,ob.UTCDateTime(time+time_window))
            phase_array=np.array([0,0.0])
            try:
                print(arrival[0].data.shape)
            except:
                print("index out of range")
                break
            if arrival[0].data.shape==(361,):
                if wf_train==[]:
                    wf_train=np.array(arrival[0].data)
                else:
                    wf_train=np.vstack([wf_train,arrival[0].data])
                print(wf_train.shape)
                if(phase_train==[]):
                    phase_train=phase_array
                    print("phase_train empty")
                else:
                    phase_train=np.vstack([phase_train,phase_array])
            else:
                print("shape wrong!")
                print(arrival[0].data.shape)


            

        print(str((float(i)/float(num_samples))*100.0)+"%")
        i+=1
phase_train=np.array(phase_train)
print("phase_train shape")
print(phase_train.shape)
np.save("phase_array.npy",phase_train)
np.save("wf_array.npy",wf_train)
model = model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(units = 64,input_shape=(wf_train[0].shape)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(activation='softmax',units=2)
])
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print(model.summary())
print(wf_train.shape)
print(phase_train)
model.fit(wf_train,(phase_train),epochs=10)
model.evaluate(wf_train,phase_train)
