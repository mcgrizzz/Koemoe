import numpy as np
import math
import torch

import shutil
import argparse
import soundfile as sf

from tqdm import tqdm
from pathlib import Path

from dataclasses import dataclass, field

input_file = Path("X:/ML/Datasets/koe/video/Frieren_S01E01.mkv")

temp_dir = input_file.parent / "temp"
temp_dir.mkdir(exist_ok=True)

outputs_dir = input_file.parent / "outputs"
outputs_dir.mkdir(exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_path = Path("H:/Documents/Dev/ML/Koe.moe/checkpoints/latest.pt")

model = torch.load(model_path)
model.to(device)
model.eval()

bs = 128 if device.type != "cpu" else 32

sr = 16000
len_sec = 6
len_samples = len_sec*sr

classes = 2
time_steps = 30
samples_per_segment = len_samples/time_steps

def map_to_range(value, in_min, in_max, out_max=1.0, out_min=0):
    return out_min + ((value - in_min)/(in_max - in_min))*(out_max - out_min)

def samples_to_t(samples):
    return samples/sr

@dataclass
class LabelData:
    name: str
    threshold: float
    padding: float
    smooth: bool = False
    smooth_n: int = 30 #in terms of time steps
    smooth_und_weight: float = 1.0
    smooth_over_weight: float = 1.0
    relative_to: int = -1
    verbose: bool = False
    events: list[tuple] = field(default_factory=list)

events = {"Speech": [], "OPED": []}

class_map = {
    0: LabelData("Speech", .51, [.75, .75], relative_to=1,verbose=False),
    1: LabelData("OPED", .25, [.3, .3], relative_to=0, smooth=True, smooth_n=60, verbose=False)
}

for c in range(classes):
    min_val = np.min(steps[:, c*3])
    max_val = np.max(steps[:, c*3])
    
    steps[:, c*3] = (steps[:, c*3] - min_val)/(max_val - min_val)
    
    label_class = class_map[c]
    thresh = label_class.threshold
    smooth_terms = label_class.smooth_n
    
    
    for i in range(len(steps)):
        step = steps[i]
        step_samples_offset = i*samples_per_segment
        valid = step[c*3]
        
        if label_class.relative_to > -1:
            valid = max(0, (valid - step[label_class.relative_to*3]))
            
        #Smoothing, constantly changing/playing with
        if label_class.smooth:
            last_n = []
            front = [] #n/2 terms before time_step
            back = [] #n/2 terms after time_step
            
            term_length = smooth_terms/2
            while (i - term_length < 0 or i + term_length >= len(steps)):
                term_length -= 1
                
            start = int(i - term_length)
            end = int(i + term_length)
            
            if start < i:
                x = steps[start:i, c*3]
                front += x.reshape((i - start)).tolist()
            if end > (i+1):
                x = steps[(i+1):end, c*3]
                back += x.reshape((end - (i + 1))).tolist()
                
            last_n += front
            last_n += [valid]
            last_n += back
            if len(last_n) > 1:
                vals = np.array(last_n)
                
                under = vals < thresh
                over = vals >= thresh
                
                vals[under] = vals[under]*label_class.smooth_und_weight
                vals[over] = vals[over]*label_class.smooth_over_weight
                x = len(last_n)/2 - np.abs(np.linspace(-1*int(len(last_n)/2), int(len(last_n)/2), vals.shape[0]))
                x = x**.5
                
                if x.max() == x.min():
                    x = np.ones(x.shape)
                    _x = np.ones(x.shape)
                else:
                    x = (x - x.min()) / (x.max() - x.min())
                    _x = x.max() - x
                
                val_over = (vals[over]*x[over]).sum()
                val_under = (vals[under]*_x[under]).sum()
                new_valid = max(0, val_over + val_under)
                valid = min(1, new_valid)
        
        start = step[1 + c*3]
        stop = step[2 + c*3]
        
        if valid >= thresh and start < stop:
            start_time = step_samples_offset + start*samples_per_segment
            stop_time = step_samples_offset + stop*samples_per_segment
            label_class.events.append((valid, start_time, stop_time))
        if label_class.verbose: print(f'{valid}')

for idx, label_class in class_map.items():
    for i in range(0, len(label_class.events)):
        curr = label_class.events[i]
        new_start = max(0, curr[1] - sr*label_class.padding[0]) 
        new_stop = min(y.shape[0] - 1, curr[1] + sr*label_class.padding[1])
        label_class.events[i] = (curr[0], new_start, new_stop)
        
#Otherwise the subsequent clip concatentation is very slow
smoothing = .1
for idx, label_class in class_map.items():
    smoothed_events = []
    previous_pointer = 0
    for i in range(1, len(label_class.events)):
        prev = label_class.events[previous_pointer]
        curr = label_class.events[i]
        if curr[1] - prev[2] <= smoothing:
            label_class.events[i] = (curr[0], prev[1], curr[2])
            label_class.events[previous_pointer] = None
        previous_pointer = i
    label_class.events = list(filter(lambda x: x, label_class.events))

sampled_idx = np.array([]) #build a list of samples that we included, so we can correctly build the oped included version without duplication
sr_correction = o_sr/sr
all_speech = np.array([])
speech_class = class_map[0]
for i in range(0, len(speech_class.events)): 
    clip_start = speech_class.events[i][1]*sr_correction
    clip_stop = speech_class.events[i][2]*sr_correction
    clip = o[int(clip_start):(int(clip_stop)+1)]
    
    all_samples = np.linspace(int(clip_start), int(clip_stop), num=(int(clip_stop) - int(clip_start) + 1), dtype=np.uint32)
    sampled_idx = np.concatenate((sampled_idx, all_samples))
    
    all_speech = np.concatenate((all_speech, clip))

sf.write(output_file, all_speech, o_sr)

op = np.array([])
ed = np.array([])
oped_class = class_map[1]
for i in range(0, len(oped_class.events)):
    clip_start = oped_class.events[i][1]*sr_correction
    clip_stop = oped_class.events[i][2]*sr_correction
    clip = o[int(clip_start):int(clip_stop)+1]
    
    all_samples = np.linspace(int(clip_start), int(clip_stop), num=(int(clip_stop) - int(clip_start) + 1), dtype=np.uint32)
    sampled_idx = np.concatenate((sampled_idx, all_samples))
    
    if clip_start >= (y.shape[0]/2):
        ed = np.concatenate((ed, clip))
    else:
        op = np.concatenate((op, clip))


if op.shape[0] > 0:
    sf.write(output_op, op, o_sr)
    
if ed.shape[0] > 0:
    sf.write(output_ed, ed, o_sr)

all = np.array([])

def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

sampled_idx = np.unique(sampled_idx)
np.sort(sampled_idx)

sampled_idx = consecutive(sampled_idx)

for i in range(len(sampled_idx)):
    segment = sampled_idx[i]
    start = int(segment[0])
    stop = int(segment[-1])
    all = np.concatenate((all, o[start:stop + 1]))

sf.write(output_file_all, all, o_sr)
shutil.rmtree(temp_dir)


