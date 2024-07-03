import numpy as np
import torch

import shutil
import argparse

from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass, field

from sample import Sample
from segment import Segments

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

frieren = Sample(input_file=input_file, output_dir=outputs_dir, temp_dir=temp_dir, include_op=True, include_ed=True, target_sr=sr)
print("Generating Audio File...")
frieren.generate_audio()
print("Loading Audio File...")
frieren.load_audio(sr, len_samples)
print("Generating Model Inputs...")
frieren.generate_inputs(bs, device)
print("Running Inference...")
outputs = frieren.infer(model)

segments = Segments(outputs)
print("Processing Outputs..")
processed = segments.get_segments(len_samples, *frieren.get_args())
print("Saving file..")
frieren.save_output(processed)


