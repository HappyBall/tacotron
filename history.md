# A (Heavily Documented) TensorFlow Implementation of Tacotron: A Fully End-to-End Text-To-Speech Synthesis Model

## **Major History**
  * June 21, 2017. Fourth draft. 
    * **I've updated the code for TF 1.1 to TF 1.2.** Turns out that TF 1.2 has a new api for attention wrapper and more detailed options.
    * I've added a sanity check option to the `hyperparams.py` file. Basically, it's purpose is to find out if our model is able to learn a handful of training data wihtout caring about generalization. For that, the training was done on a single mini-batch (32 samples) over and over again, and sample generation was based on the same text. I observed a quite smooth training curve for as below, and after around 18K global steps it started to generate recognizable sounds. The sample results after 36K steps are available in the `logdir_s` folder. It took around seven hours on a single gtx 1080. The pretrained files can be downloaded from [here](https://u42868014.dl.dropboxusercontent.com/u/42868014/tacotron/logdir_s.zip). The training curve looks like this.

<img src="fig/mean_loss.png">

  * June 4, 2017. Third draft. 
    * Some people reported they gained promising results, based on my code. Among them are, [@ggsonic](https://www.github.com/ggsonic), [@chief7](https://www.github.com/chief7). To check relevant discussions, see this [discussion](https://www.github.com/Kyubyong/tacotron/issues/30), or their repo. 
    * According @ggsonic, instance normalization worked better than batch normalization.
    * @chief7 trained on pavoque data, a German corpus spoken by a single male actor. He said that instance normalization and zero-masking are good choices.
    * Yuxuan, the first author of the paer, advised me to do sanity-check first with small data, and to adjust hyperparemters since our dataset is different from his. I really appreciate his tips, and hope this would help you.
    * [Alex's repo](https://github.com/barronalex/Tacotron), which is another implementation of Tacotron, seems to be successful in getting promising results with some small dataset. He's working on a big one.
  * June 2, 2017. 
    * Added `train_multiple_gpus.py` for multiple GPUs.
  * June 1, 2017. Second draft. 
    * I corrected some mistakes with the help of several contributors (THANKS!), and re-factored source codes so that they are more readable and modular. So far, I couldn't get any promising results.
  * May 17, 2017. First draft. 
    * You can run it following the steps below, but good results are not guaranteed. I'll be working on debugging this weekend. (**Code reviews and/or contributions are more than welcome!**)

## Requirements
  * NumPy >= 1.11.1
  * TensorFlow == 1.2
  * librosa
  * tqdm

## Data
Since the [original paper](https://arxiv.org/abs/1703.10135) was based on their internal data, I use a freely available one, instead.

[The World English Bible](https://en.wikipedia.org/wiki/World_English_Bible) is a public domain update of the American Standard Version of 1901 into modern English. Its text and audio recordings are freely available [here](http://www.audiotreasure.com/webindex.htm). Unfortunately, however, each of the audio files matches a chapter, not a verse, so is too long for many machine learning tasks. I had someone slice them by verse manually. You can download [the audio data](https://dl.dropboxusercontent.com/u/42868014/WEB.zip) and its [text](https://dl.dropboxusercontent.com/u/42868014/text.csv) from my dropbox.



## File description
  * `hyperparams.py` includes all hyper parameters that are needed.
  * `prepare_pavoque.py` creates sliced sound files from raw sound data, and constructs necessary information.
  * `prepro.py` loads vocabulary, training/evaluation data.
  * `data_load.py` loads data and put them in queues so multiple mini-bach data are generated in parallel.
  * `utils.py` has several custom operational functions.
  * `modules.py` contains building blocks for encoding/decoding networks.
  * `networks.py` has three core networks, that is, encoding, decoding, and postprocessing network.
  * `train.py` is for training.
  * `eval.py` is for sample synthesis.
  

## Training
  * STEP 1. Adjust hyper parameters in `hyperparams.py` if necessary.
  * STEP 2. Download and extract [the audio data](https://dl.dropboxusercontent.com/u/42868014/WEB.zip) and its [text](https://dl.dropboxusercontent.com/u/42868014/text.csv).
  * STEP 3. Run `train.py`. or `train_multi_gpus.py` if you have more than one gpu.

## Sample Synthesis
  * Run `eval.py` to get samples.

### Acknowledgements
I would like to show my respect to Dave, the host of www.audiotreasure.com and the reader of the audio files.
