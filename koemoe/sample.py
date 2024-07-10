import soundfile as sf
import numpy as np
from utils import *
import patch_ffprobe
import ffprobe3
import librosa
import torch
import math

from ffprobe3 import FFaudioStream
from pathlib import Path
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
    output_format: str
    temp_dir: Path
    include_op: bool = False
    include_ed: bool = False
    target_sr: int = 1600
    sample_data: SampleData = None
    
    def __post_init__(self):
        self.name = self.input_file.stem

        self.output_file = self.output_dir / self.output_format.replace("$name$", self.name)
        self.audio_file = self.temp_dir / (self.name + '.wav')
        
    def generate_audio(self):
        codec_map = { "mp3": "mp3", "opus": "opus", "vorbis": "ogg"}
        # "aac": "m4a" cannot load m4a downstream
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
        
        audio_stream: FFaudioStream = ffprobe_output.audio[audio_index]
        ext = "flac" #default to flac (free lossless audio codec) if no other supported files
        if audio_stream.codec_name in codec_map.keys():
            ext = codec_map[audio_stream.codec_name]
        
        self.audio_file = self.temp_dir / (self.name + '.' + ext)
        
        if not self.audio_file.exists():
            args = {"map":("0:a:" + str(audio_index))}
            ffmpeg = (
                FFmpeg()
                .input(str(self.input_file))
                .option("vn")
                .output(
                    self.audio_file,
                    **args
                )
            )
            try:
                ffmpeg.execute()
            except FFmpegError as exception:
                print("- Message from ffmpeg:", exception.message)
                print("- Arguments to execute ffmpeg:", " ".join(exception.arguments))
    
    def load_audio(self, target_sr, sample_len):
        
        o_sr = librosa.get_samplerate(self.audio_file)
        frame_length = (4096)
        hop_length = (2048)
        
        stream = librosa.stream(self.audio_file, block_length=256, frame_length=frame_length, hop_length=hop_length)
        
        segments = list(stream)
        o = np.array([])
        load_progress: TieredCounter = manager.counter(level=2, keep_children=False, desc="Loading Audio:", unit="blocks", total=len(segments))
        for s in segments:
            o = np.concatenate((o, s))
            load_progress.update()
            
        #o,o_sr = librosa.load(self.audio_file, sr=None)
        y = librosa.resample(o, orig_sr=o_sr, target_sr=target_sr)

        #pad end to get non-fractional number of clips
        num_clips = int(math.ceil(y.shape[0]/sample_len))
        missing_samples = num_clips*sample_len - y.shape[0]

        zeros = np.zeros(missing_samples)
        y = np.append(y, zeros, axis=0)

        #reshape into clip length
        _y = y.reshape((num_clips, sample_len))
        
        self.sample_data = SampleData(o, o_sr, y, _y)
    
    def generate_inputs(self, bs, device):
        _y = self.sample_data._y
        inputs = []
        
        input_progress: TieredCounter = manager.counter(level=2, keep_children=False, desc="Generating Inputs:", unit="segments", total=_y.shape[0])
        
        for i in range(_y.shape[0]):
            melspec = librosa.feature.melspectrogram(y=_y[i], sr=self.target_sr, hop_length=160)
            melspec = librosa.power_to_db(melspec, ref=np.max)
            inputs.append(melspec)
            input_progress.update()

        inputs = np.array(inputs).astype(np.float32)
        inputs = torch.from_numpy(np.array([inputs])).to(device)
        inputs = inputs.permute(1, 0, 2, 3)

        batches = int(math.ceil(inputs.shape[0]/bs))
        
        self.bs = bs
        self.sample_data.inputs = inputs
        self.sample_data.batches = batches

    def infer(self, model):
        batches = self.sample_data.batches
        inputs = self.sample_data.inputs
        outputs = []
        
        infer_progress: TieredCounter = manager.counter(level=2, keep_children=False, desc="Inferring:", unit="batches", total=batches)
        
        with torch.inference_mode():
            for b in range(batches):
                start = b*self.bs
                end = (b+1)*self.bs if b != (batches - 1) else inputs.shape[0]
                batch = inputs[start:end]
                
                with nowarning():
                    outputs += model(batch)
                infer_progress.update()

        for i in range(len(outputs)):
            outputs[i] = outputs[i].cpu().numpy()
            
        outputs = np.array(outputs)
        steps = outputs.reshape((outputs.shape[0]*outputs.shape[1], outputs.shape[2])) #flatten since we do not need the extra clip dimension
        return steps
    
    #sample_length, ori_length: int, include_op=False, include_ed=False
    def get_args(self):
        return [self.sample_data.y.shape[0], self.include_op, self.include_ed]
    
    def save_output(self, audio_segments: np.ndarray):
        audio = np.array([])
        sr_correction = self.sample_data.o_sr/self.target_sr
        
        save_progress: TieredCounter = manager.counter(level=2, keep_children=False, desc="Saving:", unit="segments", total=len(audio_segments))
        
        for i in range(len(audio_segments)):
            segment = audio_segments[i]
            start = int(segment[0]*sr_correction)
            stop = int(segment[-1]*sr_correction)
            audio = np.concatenate((audio, self.sample_data.o[start:stop + 1]))
            save_progress.update()
            
        sf.write(self.output_file, audio, self.sample_data.o_sr)
        return audio.shape[0]/self.sample_data.o_sr

    def reset(self):
        self.sample_data.o = None
        self.sample_data.y = None
        self.sample_data._y = None
        self.sample_data = None
        