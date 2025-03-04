#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
"""
import os
from dataclasses import dataclass
from typing import Optional
from sed_demo.utils import load_csv_labels
from sed_demo.models import Cnn9_GMP_64x64, Cnn14_pruned
from sed_demo.audio_loop import AsynchAudioInputStream
from sed_demo.inference import AudioModelInference, PredictionTracker
import torch

from threading import Thread

import os
import numpy as np


class DemoApp():
    """
    This class extends the Tk frontend with the functionality to run the audio
    detection demo
    """

    BG_COLOR = "#fff8fa"
    BUTTON_COLOR = "#ffcc99"
    BAR_COLOR = "#ffcc99"
    def __init__(
            self,
            all_audioset_labels, tracked_labels=None,
            samplerate=32000, audio_chunk_length=1024, ringbuffer_length=40000,
            model_winsize=1024, stft_hopsize=512, stft_window="hann", n_mels=64,
            mel_fmin=50, mel_fmax=14000, top_k=5):
        """
        """
        
        #
        self.is_running = True
        self.audiostream = AsynchAudioInputStream(
            samplerate, audio_chunk_length, ringbuffer_length)
        # 2. DL model to predict tags from ring buffer
        num_audioset_classes = len(all_audioset_labels)
        self.model = Cnn14_pruned(sample_rate=32000, window_size=1024,hop_size=320,mel_bins=64,fmin=50,fmax=14000,classes_num=527,p1=0,p2=0,p3=0,p4=0,p5=0,p6=0,p7=0.5,p8=0.5,p9=0.5,p10=0.5,p11=0.5,p12=0.5)#Cnn9_GMP_64x64(num_audioset_classes)
        checkpoint = torch.load(model_path,
                                map_location=lambda storage, loc: storage)
        self.model.load_state_dict(checkpoint)#["model"])
        self.inference = AudioModelInference(
            self.model, model_winsize, stft_hopsize, samplerate, stft_window,
            n_mels, mel_fmin, mel_fmax)
        self.tracker = PredictionTracker(audioset_labels,
                                         allow_list=tracked_labels)
        #
        self.top_k = top_k
        self.thread = None
        # handle when user closes window
        # self.protocol("WM_DELETE_WINDOW", self.exit_demo)
        # self.info()
    # def info(self):
    #     self.model
    #     summary(self.model,(64000,),device='cpu')

    def inference_loop(self):
        """
        """
        while self.is_running:
            dl_inference = self.inference(self.audiostream.read())
            top_preds = self.tracker(dl_inference, self.top_k)
            print(top_preds)
            #
            # for label, bar, (clsname, pval) in zip(
            #         self.sound_labels, self.confidence_bars, top_preds):
            #     label["text"] = clsname
            #     bar["value"] = pval


    def is_running(self):
        """
        """
        return self.is_running

    def start(self):
        """
        """
        self.audiostream.start()
        self.thread = Thread(target=self.inference_loop)
        self.thread.start()  # will end automatically if is_running=False

    def stop(self):
        """
        """
        self.audiostream.stop()

    def exit_demo(self):
        """
        """
        print("Exiting...")
        if self.is_running:
            self.toggle_start()
        self.audiostream.terminate()
        self.destroy()




# config
audioset_labels_path = os.path.join("config", "audioset_labels.csv")
domestic_labels_path = os.path.join("config", "domestic_labels.csv")
model_path = os.path.join(
    "models", "checkpoint_closeto_.44.pt")
#"Cnn9_GMP_64x64_300000_iterations_mAP=0.37.pth")
samplerate = 32000
audio_chunk_length = 1024
ringbuffer_length = int(samplerate * 2)  # 62 * 1024  # around 2s

model_winsize = 1024
stft_hopsize = 320
stft_window = "hann"
n_mels = 64
mel_fmin = 50
mel_fmax = 14000

top_k = 8 # change to display top few predictions


# 0. Load AudioSet and allowed labels
_, _, audioset_labels = load_csv_labels(audioset_labels_path)
_, _, domestic_labels = load_csv_labels(domestic_labels_path)




demo = DemoApp(audioset_labels, domestic_labels,
               samplerate, audio_chunk_length, ringbuffer_length,
               model_winsize, stft_hopsize, stft_window, n_mels,
               mel_fmin, mel_fmax, top_k)

demo.inference_loop()


# breakpoint()






# ##############################################################################
# # OMEGACONF
# ##############################################################################
@dataclass
class ConfDef:
    """
    :cvar SAMPLERATE: Imported samplerate (regardless of the original one)
    :cvar FIG_SAVEPATH: If not given, figure will be shown on screen.
    """
    AUDIO_PATH: str =  os.path.join("datasets", "R9_ZSCveAHg_7s.wav")
    LABELS_PATH: str = os.path.join("datasets", "class_labels_indices.csv")
    MODEL_PATH: str = os.path.join("models", "checkpoint_closeto_.44.pt")
    #
    DEVICE: str = "cpu"
    SAMPLERATE: int = 32000
    TOP_K: int = 5
    FIG_SAVEPATH: Optional[str] = None


# ##############################################################################
# # HELPERS
# ##############################################################################


# # ##############################################################################
# # # MAIN ROUTINE
# # ##############################################################################
# if __name__ == '__main__':
#     # Load conf
#     CONF = OmegaConf.structured(ConfDef())
#     cli_conf = OmegaConf.from_cli()
#     CONF = OmegaConf.merge(CONF, cli_conf)
#     print("\n\nCONFIGURATION:")
#     print(OmegaConf.to_yaml(CONF), end="\n\n\n")
