#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import sys
import json
from .lib.utils import load_csv_labels
from .lib.models import Cnn14_pruned
from .lib.inference import AudioModelInference, PredictionTracker
from torch import load as loadModel
from torch.cuda import is_available as hasCUDA
from librosa.core import load as loadAudio
from typer import Argument, Option, BadParameter
from typing_extensions import Annotated

samplerate = 32000
model_winsize = 1024
stft_hopsize = 320
stft_window = "hann"
n_mels = 64
mel_fmin = 50
mel_fmax = 14000

default_labels_path = os.path.join(os.path.dirname(__file__), "config", "audioset_labels.csv")
default_top_k = 10
default_checkpoint_path = os.path.join(os.path.dirname(__file__), "models", "checkpoint_closeto_.44.pt")

def check_path(value: str):
    if not os.path.isfile(value):
      raise BadParameter("File does not exist")
    return value

def check_top_k(value: int):
    if value < 1:
        raise BadParameter("Top K must be 1 or greater")
    return value

def predict(
    audio_path: Annotated[str, Argument(help="Path to the audio file", callback=check_path)] = "",
    top_k: Annotated[int, Option(help="Number of classes to return", callback=check_top_k)] = default_top_k,
    checkpoint_path: Annotated[str, Option(help="Path of checkpoint", callback=check_path)] = default_checkpoint_path,
    audioset_labels_path: Annotated[str, Option(help="Path of labels", callback=check_path)] = default_labels_path
):

  print(f"Loading labels from {audioset_labels_path}", file=sys.stderr)
  _, _, audioset_labels = load_csv_labels(audioset_labels_path)
  num_audioset_classes = len(audioset_labels)

  print(f"Loading model", file=sys.stderr)
  model = Cnn14_pruned(
    samplerate, model_winsize, stft_hopsize, n_mels,
    mel_fmin, mel_fmax, num_audioset_classes,
    p1=0,p2=0,p3=0,p4=0,p5=0,p6=0,p7=0.5,p8=0.5,p9=0.5,p10=0.5,p11=0.5,p12=0.5
  )

  print(f"Loading checkpoint from {checkpoint_path}", file=sys.stderr)
  checkpoint = loadModel(checkpoint_path, map_location=lambda storage, loc: storage)
  model.load_state_dict(checkpoint)

  print(f"Loading audio from {audio_path}", file=sys.stderr)
  (audio, _) = loadAudio(audio_path, sr=samplerate, mono=True)

  device = "cuda" if hasCUDA() else "cpu"
  print(f"Running inference on {device}", file=sys.stderr)
  inference = AudioModelInference(
      model, model_winsize, stft_hopsize, samplerate,
      stft_window, n_mels, mel_fmin, mel_fmax)
  tracker = PredictionTracker(audioset_labels)
  dl_inference = inference(audio, device)
  top_preds = tracker(dl_inference, top_k)
  print(json.dumps(top_preds))
