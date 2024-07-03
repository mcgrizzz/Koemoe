import numpy as np

from tqdm import tqdm
from dataclasses import dataclass, field

@dataclass
class LabelConfig:
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

def consecutive(data, stepsize=1):
        return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

@dataclass
class Segments:
    steps: np.ndarray
    classes: int = 2
    time_steps: int = 30
    
    def __post_init__(self):
        
        self.class_map = {
            0: LabelConfig("Speech", .51, [.75, .75], relative_to=1,verbose=False),
            1: LabelConfig("OPED", .25, [.3, .3], relative_to=0, smooth=True, smooth_n=60, verbose=False)
        }
    
    def get_segments(self, sample_length, ori_length: int, include_op=False, include_ed=False):
        
        classes = 2
        time_steps = 30
        samples_per_segment = sample_length/time_steps
        
        print("Processing Inputs...")
        for c in range(classes):
            print(f' -- Class: {self.class_map[c].name} -- ')
            min_val = np.min(self.steps[:, c*3])
            max_val = np.max(self.steps[:, c*3])
            
            self.steps[:, c*3] = (self.steps[:, c*3] - min_val)/(max_val - min_val)
            
            label_conf = self.class_map[c]
            thresh = label_conf.threshold
            smooth_terms = label_conf.smooth_n
            
            
            threshold_progress = tqdm(range(len(self.steps)))
            for i in threshold_progress:
                step = self.steps[i]
                step_samples_offset = i*samples_per_segment
                valid = step[c*3]
                
                if label_conf.relative_to > -1:
                    valid = max(0, (valid - step[label_conf.relative_to*3]))
                    
                #Smoothing, constantly changing/playing with
                if label_conf.smooth:
                    last_n = []
                    front = [] #n/2 terms before time_step
                    back = [] #n/2 terms after time_step
                    
                    term_length = smooth_terms/2
                    while (i - term_length < 0 or i + term_length >= len(self.steps)):
                        term_length -= 1
                        
                    start = int(i - term_length)
                    end = int(i + term_length)
                    
                    if start < i:
                        x = self.steps[start:i, c*3]
                        front += x.reshape((i - start)).tolist()
                    if end > (i+1):
                        x = self.steps[(i+1):end, c*3]
                        back += x.reshape((end - (i + 1))).tolist()
                        
                    last_n += front
                    last_n += [valid]
                    last_n += back
                    
                    if len(last_n) > 1:
                        vals = np.array(last_n)
                        
                        under = vals < thresh
                        over = vals >= thresh
                        
                        vals[under] = vals[under]*label_conf.smooth_und_weight
                        vals[over] = vals[over]*label_conf.smooth_over_weight
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
                    label_conf.events.append((valid, start_time, stop_time))
                if label_conf.verbose: print(f'{valid}')

        #Add in padding
        for idx, label_conf in self.class_map.items():
            for i in range(0, len(label_conf.events)):
                curr = label_conf.events[i]
                new_start = max(0, curr[1] - sample_length*label_conf.padding[0]) 
                new_stop = min(ori_length - 1, curr[1] + sample_length*label_conf.padding[1])
                label_conf.events[i] = (curr[0], new_start, new_stop)
                
        #Smooth close segments together, otherwise the subsequent clip concatentation is very slow
        smoothing = .1
        for label_conf in self.class_map.values():
            previous_pointer = 0
            for i in range(1, len(label_conf.events)):
                prev = label_conf.events[previous_pointer]
                curr = label_conf.events[i]
                if curr[1] - prev[2] <= smoothing:
                    label_conf.events[i] = (curr[0], prev[1], curr[2])
                    label_conf.events[previous_pointer] = None
                previous_pointer = i
            label_conf.events = list(filter(lambda x: x, label_conf.events))

        sampled_idx = np.array([]) #build a list of samples that we included, so we can correctly build the oped included version without duplication
        speech_class = self.class_map[0]
        for i in range(0, len(speech_class.events)): 
            clip_start = speech_class.events[i][1]
            clip_stop = speech_class.events[i][2]
            
            all_samples = np.linspace(int(clip_start), int(clip_stop), num=(int(clip_stop) - int(clip_start) + 1), dtype=np.uint32)
            sampled_idx = np.concatenate((sampled_idx, all_samples))
        
        oped_class = self.class_map[1]
        for i in range(0, len(oped_class.events)):
            clip_start = oped_class.events[i][1]
            clip_stop = oped_class.events[i][2]
            
            all_samples = np.linspace(int(clip_start), int(clip_stop), num=(int(clip_stop) - int(clip_start) + 1), dtype=np.uint32)
            
            if clip_start >= (ori_length/2)*samples_per_segment:
                if include_ed: sampled_idx = np.concatenate((sampled_idx, all_samples))
            else:
                if include_op: sampled_idx = np.concatenate((sampled_idx, all_samples))

        sampled_idx = np.unique(sampled_idx)
        np.sort(sampled_idx)

        sampled_idx = consecutive(sampled_idx)
        return sampled_idx  