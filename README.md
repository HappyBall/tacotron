# Tacotron for research on Chinese speech synthesis and Taiwanese speech synthesis from Chinese input text sequence with different granularities

Train the Tacotron speech synthesis model to synthesize Chinese or Taiwanese
speech conditioned on Chinese input text sequence with different granularities

## Prerequisites
1. Python packages:
    - Python 3.4 or higher
    - Tensorflow r1.8 or higher
    - Numpy

2. Clone this repository:
```shell=
 https://github.com/HappyBall/tacotron.git
```

## Dataset

1. Lecture record Chinese dataset [Download](http://speech.ee.ntu.edu.tw/~yangchiyi/lecture_tts_data.tgz)

2. DaAiSermon Taiwanese dataset [Download](http://speech.ee.ntu.edu.tw/~yangchiyi/DaAiSermon.tgz)

## Usage

After downloading the correspond dataset, you can directly train Tacotron model to synthesize Chinese or Taiwanese speech.

### Training

Set up the correct path of the dataset and other hyperparameters in `hyperparams.py`.

Run:
`python train.py --keep_train False`

Parameter `--keep_train` determines either start a new training or continue
training with the existed model which the path should be correctly set up in `hyperparams.py`.

### Synthesis

1. Set up the correct path of the existed model in `hyperparams.py`.

2. Add Chinese input sequences you want to synthesize into
   `test_sentences.txt`.

Run:
`python synthesize.py`

### Evaluate

Set up the output file path in `evaluate.py`.

Run:
`python evaluate.py`

The program will automatically calculate the mean square error between the mel scale spectrogram of synthesized speech and the ground truth then output as a text file.

### Hyperparameters of the hyperparams.py
`--data`: the path of the data directory which contains the wav files  
`--prepro_path`: the path of the preprocessed data directory  
`--test_data`: the path of the text file which contains input text sequences to synthesize speech  
`--logdir`: the path of the directory to save or load models  
`--logfile`: the path of the training log file  
`--sampledir`: the path of the directory to save speech files when synthesizing  
`--input_mode`: mode of the input granularity (word, bopomofo, pinyin, phoneme)  
`--withtone`: input with tone or not, only bopomofo or pinyin input mode use this hyperparameter (True, False)  
`--n_iter`: iteration number of Griffin Lim algorithm  
`--guided_attention`: use guided attention or not when training (True, False)  
`--schedule_prob`: probability of schedule sampling using the ground truth as input  
`--lr`: initial learning rate  


## Files in this project
`data_load.py`: data loader for training data and testing data  
`evaluate.py`: calculate the mean square error between the mel scale spectrogram of synthesized speech and the ground truth from testing data  
`graph.py`: define model graph  
`hyperparams.py`: set up training hyperparameters and directory for saving models  
`modules.py`: define modules like gru, convolution layers ...etc  
`networks.py`: define networks like encoder, decoder ...etc  
`prepro.py`: preprocess data  
`synthesize.py`: synthesize speech conditioned on input text sequences in the test sentence file  
`train.py`: train models  
