import torch
import sys
import os

import glob
import shutil
import argparse

import enlighten
from enlighten._util import format_time
from pathlib import Path
from dataclasses import dataclass, field

from utils import *
from sample import Sample
from segment import Segments
from os import environ
from tempfile import TemporaryDirectory

status:enlighten.StatusBar = manager.status_bar(status_format=u'[Koemoe]{fill}{stage}{fill}{elapsed}',
                                    color='bold_underline_mistyrose_on_firebrick1',
                                    justify=enlighten.Justify.CENTER, stage='Initializing'.upper(),
                                    autorefresh=True, min_delta=0.5)

parser = argparse.ArgumentParser(prog="Koemoe")
parser.add_argument('input', type=Path, help="The input file or directory")
parser.add_argument('--include-op', "-io", action='store_true', help="include the OP in the condensed file")
parser.add_argument('--include-ed', "-ie", action='store_true', help="include the ED in the condensed file")
parser.add_argument('--output-dir', "-o", type=Path, help="change the output directory")
parser.add_argument('--output-format', "-f", type=str, default="$name$_condensed.wav", help="change the output file name (default: $name$_condensed.wav)")
#tested output_formats = ["wav", "flac", "mp3", NO:"ogg"]

args = parser.parse_args()

input_files = []
input: Path = args.input

sr = 16000
len_sec = 6
len_samples = len_sec*sr

input_total_time: float = 0
output_total_time: float = 0

sample_map = {}

if input.is_file():
    input_files.append(input)
else:
    mkvs = [p for p in Path(input).rglob("*") if p.suffix.lower() == ".mkv"]
    mp4s = [p for p in Path(input).rglob("*") if p.suffix.lower() == ".mp4"]
    input_files += mkvs
    input_files += mp4s

if not len(input_files):
    status.update(stage="Error".upper(), force=True)
    print("ERROR: No input files found")
    sys.exit()
    
temp_dir = input_files[0].parent / "temp"
if temp_dir.exists():
    print(f'INFO: {str(temp_dir)} Exists')
else:
    temp_dir.mkdir(exist_ok=True)
    print(f'INFO: {str(temp_dir)} Created')

if not args.output_dir:
    output_dir = input_files[0].parent / "outputs"
else:
    output_dir = args.output_dir

if output_dir.exists():
    print(f'INFO: {str(output_dir)} Exists')
else:
    output_dir.mkdir(exist_ok=True)
    print(f'INFO: {str(output_dir)} Created')

status.update(stage="Loading Model".upper())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu") #test cpu inference
print(f'INFO: Finding Newest Model')
newest = get_latest_model()
if newest == "None":
    status.update(stage="Error".upper(), force=True)
    print("ERROR: No model files found. Please check https://github.com/mcgrizzz/Koemoe/releases for the latest model")
    sys.exit()
print(f'INFO: Loading \'{newest}\' using \'{device}\' ...')
model_path = Path("model/") / newest
model = torch.load(model_path, map_location=device)
#model.to(device)
model.eval()
print(f'INFO: Model Loaded')

bs = 128 if device.type != "cpu" else 32

with TemporaryDirectory() as tmp_cache:
    print(f'INFO: Temporary Cache Dir {tmp_cache}')
    environ["NUMBA_CACHE_DIR"] = tmp_cache

status.update(stage="Generating Audio Files".upper())
sub_progress: TieredCounter = manager.counter(desc="Generating Audio:", total=len(input_files))
for input_file in input_files:
    cmd_args = { "output_dir":output_dir, "output_format":args.output_format, "include_op":args.include_op, "include_ed": args.include_ed}
    sample = Sample(input_file=input_file, temp_dir=temp_dir, target_sr=sr, **cmd_args)
    sample.generate_audio()
    print(f'INFO: Audio Generated and saved to "{str(sample.audio_file)}"')
    sample_map[input_file] = sample
    sub_progress.update()

status.update(stage="Processing Files".upper())
sub_progress: TieredCounter = manager.counter(desc="Processing Samples:", total=len(input_files))
for input_file in input_files:
    sample: Sample = sample_map[input_file]
    
    sample_progress: TieredCounter = manager.counter(level=1, keep_children=False, desc=f'{sample.name}', total=5)
    print(f'INFO: -- {sample.name} -- ')
    
    sample.load_audio(sr, len_samples)
    sample_progress.update()
    print(f'INFO: Audio Loaded')
    input_total_time += sample.sample_data.o.shape[0]/sample.sample_data.o_sr
    
    #sub_progress.set_description("Generating Model Inputs")
    sample.generate_inputs(bs, device)
    sample_progress.update()
    print(f'INFO: Inputs Generated')
    
    #sub_progress.set_description("Running Inference")
    outputs = sample.infer(model)
    sample_progress.update()
    print(f'INFO: Inference Completed')
    
    #sub_progress.set_description("Processing Outputs")
    segments = Segments(outputs)
    processed = segments.get_segments(len_samples, *sample.get_args())
    sample_progress.update()
    print(f'INFO: Model Outputs Processed')
    
    #sub_progress.set_description("Saving File")
    output_time = sample.save_output(processed)
    sample.reset()
    segments.reset()
    sample_progress.update()
    sample_progress.close()
    sub_progress.update()
    print(f'INFO: Saved output to "{str(sample.output_file)}"')
    
    output_total_time += output_time
    
status.update(stage="Cleaning Up".upper(), force=True)
temps = os.listdir(temp_dir)
cleanup: TieredCounter = manager.counter(desc="Cleaning up:", total=len(temps) + 1)
for audio in temps:
    os.remove(temp_dir / audio)
    print(f'INFO: "{str(temp_dir / audio)}" Deleted')
    cleanup.update()

shutil.rmtree(temp_dir)
print(f'INFO: "{str(temp_dir)}" Deleted')
cleanup.update()

print(f'INFO: Completed')
status.update(stage="Completed".upper(), force=True)

time_elapsed = format_time(status.elapsed).split(":")
input_runtime = format_time(input_total_time).split(":")
output_runtime = format_time(output_total_time).split(":")
reduction = round(((input_total_time - output_total_time)/input_total_time)*100, 2)
print(f'=========== SUMMARY ===========')
print(f'Processed {len(input_files)} file(s)')
print(f'Compressed runtime {input_runtime[0]}m {input_runtime[1]}s -> {output_runtime[0]}m {output_runtime[1]}s ({reduction}% reduction)')
print(f'Processing took a total of {time_elapsed[0]}m {time_elapsed[1]}s')
