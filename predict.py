#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import sys
from lib.utils import load_csv_labels
from lib.models import Cnn14_pruned
from lib.inference import AudioModelInference, PredictionTracker
from torch import load as loadModel
from librosa.core import load as loadAudio
import os

audio_path = '/Users/chrisbaume/Downloads/segments/gap_07.wav'

samplerate = 32000
audio_chunk_length = 1024
ringbuffer_length = int(samplerate * 2)
model_winsize = 1024
stft_hopsize = 320
stft_window = "hann"
n_mels = 64
mel_fmin = 50
mel_fmax = 14000
top_k = 8

audioset_labels_path = os.path.join("config", "audioset_labels.csv")
_, _, audioset_labels = load_csv_labels(audioset_labels_path)
num_audioset_classes = len(audioset_labels)

print(f"Loading model", file=sys.stderr)
model = Cnn14_pruned(
  samplerate, model_winsize, stft_hopsize, n_mels,
  mel_fmin, mel_fmax, num_audioset_classes,
  p1=0,p2=0,p3=0,p4=0,p5=0,p6=0,p7=0.5,p8=0.5,p9=0.5,p10=0.5,p11=0.5,p12=0.5
)
print(f"Model loaded", file=sys.stderr)

model_path = os.path.join("models", "checkpoint_closeto_.44.pt")
print(f"Loading checkpoint from {model_path}", file=sys.stderr)
checkpoint = loadModel(model_path, map_location=lambda storage, loc: storage)
model.load_state_dict(checkpoint)
print(f"Checkpoint loaded", file=sys.stderr)

print(f"Running inference", file=sys.stderr)
inference = AudioModelInference(
    model, model_winsize, stft_hopsize, samplerate,
    stft_window, n_mels, mel_fmin, mel_fmax)
tracker = PredictionTracker(audioset_labels)

(audio, _) = loadAudio(audio_path, sr=samplerate, mono=True)
dl_inference = inference(audio)
print(f"Inference complete", file=sys.stderr)

top_preds = tracker(dl_inference, top_k)
print(top_preds)
