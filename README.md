# epanns

`epanns` is a command-line tool for categorising sound within an audio file. It is uses the [E-PANNs](https://github.com/Arshdeep-Singh-Boparai/E-PANNs) lightweight pre-trained model developed by Arshdeep Singh at the University of Surrey. Sounds are categorised using the [Google AudioSet ontology](https://research.google.com/audioset/ontology/index.html).

## Installation
Use [pipx](https://pipx.pypa.io/stable/) to install:
```
pipx install epanns
```

## Usage
```
epanns /path/to/audio.wav
```
will return the predicted categories and their probability as JSON
```
[
  [
    "Speech",
    0.7508
  ],
  [
    "Inside, small room",
    0.0186
  ],
  [
    "Computer keyboard",
    0.0145
  ]
]
```

To see the available options, run `epanns --help`

If you do not provide a checkpoint path, the [model checkpoint](https://zenodo.org/records/7939403) will be downloaded on the first run and cached for future runs.


## Acknowledgements

This software is based on the following research. Please cite these papers:

- Arshdeep Singh, Haohe Liu and Mark D PLumbley. "E-PANNS: Sound Recognition using Efficient Pre-Trained Audio Neural Networks", accepted in Internoise 2023.

- Singh, Arshdeep, and Mark D. Plumbley. "Efficient CNNs via Passive Filter Pruning." arXiv preprint arXiv:2304.02319 (2023). 

- Qiuqiang Kong, Yin Cao, Turab Iqbal, Yuxuan Wang, Wenwu Wang, Mark D. Plumbley. "PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition." arXiv preprint arXiv:1912.10211 (2019).

The research was supported by Engineering and Physical Sciences Research Council (EPSRC) Grant EP/T019751/1 “AI for Sound (AI4S)”. Project link:  https://ai4s.surrey.ac.uk/

## Related links

* https://research.google.com/audioset/dataset
* https://github.com/qiuqiangkong/audioset_tagging_cnn
* https://github.com/qiuqiangkong/panns_inference
* https://github.com/yinkalario/Sound-Event-Detection-AudioSet


## License

[MIT](https://choosealicense.com/licenses/mit/)