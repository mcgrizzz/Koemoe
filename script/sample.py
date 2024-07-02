import numpy as np
import ffprobe3
import librosa
import torch
import math

from tqdm import tqdm
from pathlib import Path
from typing import Tuple, Union
from ffmpeg import FFmpeg, FFmpegError
from dataclasses import dataclass, field

@dataclass
class SampleData:
    o: np.ndarray
    o_sr: int
    y: np.ndarray
    _y: np.ndarray
    inputs: np.ndarray = None
    batches: int = 0
    
@dataclass
class Sample:
    input_file: Path
    output_dir: Path
    temp_dir: Path
    include_op: bool = False
    include_ed: bool = False
    target_sr: int = 1600
    sample_data: SampleData = None
    
    def __post_init__(self):
        self.name = self.input_file.stem

        self.output_file = self.output_dir / (self.name + "_condensed.wav")
        self.audio_file = self.temp_dir / (self.name + '.wav')
        
    def generate_audio(self):
        if not self.audio_file.exists():
            ffprobe_output = ffprobe3.probe(str(self.input_file))    

            audio_index = 0 #default to 
            for i in range(len(ffprobe_output.audio)):
                s = ffprobe_output.audio[i]
                tags = s.parsed_json['tags']
                if "language" not in tags:
                    break
                if tags["language"] == "jpn":
                    audio_index = i
                    break

            ffmpeg = (
                FFmpeg()
                .input(str(self.input_file))
                .option("vn")
                .output(
                    self.temp_dir / (self.name + '.wav'),
                    map=["0:a:" + str(audio_index)],
                    acodec="pcm_s16le",
                )
            )
            try:
                ffmpeg.execute()
            except FFmpegError as exception:
                print("- Message from ffmpeg:", exception.message)
                print("- Arguments to execute ffmpeg:", " ".join(exception.arguments))
    
    def load_audio(self, target_sr, sample_len):
        o,o_sr = librosa.load(self.audio_file, sr=None)
        y = librosa.resample(o, orig_sr=o_sr, target_sr=target_sr)

        #pad end to get non-fractional number of clips
        num_clips = int(math.ceil(y.shape[0]/sample_len))
        missing_samples = num_clips*sample_len - y.shape[0]

        zeros = np.zeros(missing_samples)
        y = np.append(y, zeros, axis=0)

        #reshape into clip length
        _y = y.reshape((num_clips, sample_len))
        
        self.sample_data = SampleData(o, o_sr, y, _y)
    
    def generate_inputs(self, bs):
        _y = self.sample_data._y
        inputs = []
        for i in range(_y.shape[0]):
            melspec = librosa.feature.melspectrogram(y=_y[i], sr=self.target_sr, hop_length=160)
            melspec = librosa.power_to_db(melspec, ref=np.max)
            inputs.append(melspec)

        inputs = np.array(inputs).astype(np.float32)
        inputs = torch.from_numpy(np.array([inputs]))
        inputs = inputs.permute(1, 0, 2, 3)

        batches = int(math.ceil(inputs.shape[0]/bs))
        
        self.sample_data.inputs = inputs
        self.sample_data.batches = batches

    def infer(self, model, bs):
        batches = self.sample_data.batches
        inputs = self.sample_data.inputs
        print("Starting inference...")
        outputs = []
        with torch.no_grad():
            for b in tqdm(range(batches)):
                start = b*bs
                end = (b+1)*bs if b != (batches - 1) else inputs.shape[0]
                batch = inputs[start:end]
                
                outputs += model(batch)

        for i in range(len(outputs)):
            outputs[i] = outputs[i].cpu()

        outputs = np.array(outputs)
        steps = outputs.reshape((outputs.shape[0]*outputs.shape[1], outputs.shape[2])) #flatten since we do not need the extra clip dimension
        return steps