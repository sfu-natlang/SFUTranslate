<img src="docs/logo.jpg" align="left"/>
<img src="docs/natlang-logo.png" align="right" width="150"/>
<h1 align="center"> SFUTranslate </h1>
<br/><br/>

This is an academic machine translation toolkit, in which the main focus has been towards readability and changeability.
We also have tried to make the algorithms as fast as possible, but please let us know if you have any suggestions or
concerns regarding the toolkit. To get familiar with what you can do and how you can do it please read through this documentation.

To run the code you will need python 3.5+ and PyTorch 1.1+.

# Getting Started

To get started, we start with the project structure. In the highest level of the project, there are two main directories:

- `resurces`: All the necessary resources for the project are supposed to be loaded from the `SFUTranslate/resources` directory.
Please refrain from putting the resources anywhere else if you are planing to make a pull request.
 
- `translate`: All the source code files are placed under `SFUTranslate/translate` directory.
If you are using an IDE to debug or run the code, don't forget to mark this directory as your sources directory.
Otherwise you will need to have the address of `SFUTranslate/translate` directory in your `$PATH` environment variable to be able to run the code.
Another way of running the code would be to run `cd /path/to/SFUTranslate/translate && python trainer.py <path/to/config.yml>`. 
To test the trained model on the test set, you may use the `test_trained_model` script 
(e.g. `cd /path/to/SFUTranslate/translate && python test_trained_model.py <path/to/config.yml>`). 
 
The next sections will help you get more familiar with the code flow and training different models.

## Code Structure
As stated earlier, the source codes are placed in the `translate` directory (i.e. `/path/to/SFUTranslate/translate/`). 

The general structure of the modules is depicted in the following figure.

<img src="docs/SFUTranslate.svg?sanitize=True"/>

As of the current version, the `translate` package contains the following sub-packages and classes.
  - `models`
    + `general`   
    + `sts`
  - `readers`
    + `data_provider`   
    + `datasets`
    + `utils`
  - `utils`
    + `containers`   
    + `evaluation`
    + `init_nn`
    + `optimizers`
  - `configuration`
  - `extract_common_vocab`
  - `test_trained_model`
  - `trainer` the main script which loads the config file, creates the model and runs the training process. 
  You may want to start looking into this script first, to get familiar with what can be done using this toolkit.
  

## What can be put in the config file?
Here we present a complete schema for the config file containing all possible valid tags that can be put in the config file.
Please note that you may put some in and remove some from your config file, however, if the config file lacks the configurations 
that are essential to your task you will face an error indicating that the required configuration value is not presented.
  In that case, please look at the config file and put the configuration tag with your desired value in it. An example 
  config file called `nmt.yml` is already put in the `resources` directory so you can modify and use for running the 
  project. Nevertheless, you can create your own config file as a text file whose name is ending in `.yml` and put your 
  configurations in it. Here is the configuration schema:
```yamlex
debug_mode: [true/false] if true spacy tokenizer is deactivated and Multi30k dataset is automatically loaded
src_lang: the bi-letter language identifier for source langugage
tgt_lang: the bi-letter language identifier for target langugage
dataset_name: the name of the torchtext datasetname
lowercase_data: [true/false] whether the dataset setences need to be lowercased or not
pad_token: special pad token used in data transformation
bos_token: special begin of sentence token used in data transformation
eos_token: special end of sentence token used in data transformation
unk_token: special unk token used in data transformation
propn_token: special proper noun indicator used in data transformation [currently this is deactivated]
max_sequence_length: maximum train data sentence length which is considered
max_vocab_src: maximum size of source side vocabulary 
min_freq_src: minimum considrable source vocabulary, words with less freqency than this are replaced with ``unk_token``
max_vocab_tgt: maximum size of target side vocabulary
min_freq_tgt: minimum considrable target vocabulary, words with less freqency than this are replaced with ``unk_token``

emb_dropout: the droupout ratio applied to both source and target embedding layers
batch_size: the average number of words expected to be put in a batch [4000 to 5000 seem to be a reasonable defalt]
encoder_emb_size: the size of the embedding layer in encoder
encoder_hidden_size: the size of the RNN layer in encoder
encoder_layers: number of encoder RNN layers
encoder_dropout_rate: the dropout value applied to the encoder RNN
decoder_emb_size: the size of the embedding layer in decoder
decoder_hidden_size: the size of the RNN layer in decoder if this is not equal to twice the size of encoder output, a bridge network automatically mapps the encoder hidden states to the decoder hidden states.
decoder_layers: number of decoder RNN layers
decoder_dropout_rate: the dropout value applied to the decoder RNN
out_dropout: the dropout value applied to the output of attention before getting fed to the affine transformation
coverage_dropout: the droput value applied to the encoded representations before phi parameter is created 

maximum_decoding_length: maximum valid decoding length
bahdanau_attention: [true/false] if not true Loung general attention is applied 
coverage_phi_n: the N value for calculating the phi parameter in linguistic coverage calculation
coverage_required: [true/false] indicates whether coverage mechanism will be conisdered or not 
coverage_lambda: the justification coefficient for the coverage loss when its added to the NLL loss

n_epochs: number of iterations over the all of training data 
init_optim: the optimizer with which model is initialized [normally for just a few iterations] 
init_learning_rate: the inital leaning rate of init_optim 
init_epochs: the number of iterations that ``init_optim`` looks at the train data to initialize the parameters 
optim: the optimizer with which the initialized model is tranied
learning_rate: the initial leaning rate of ``optim``
learning_momentum: the momentum used in ``optim`` if it is ``SGD``
grad_clip: [true/false] indicates whether graient clipping needs to be applied to the gradients
max_grad_norm: maximum valid gradient norm, the values greater than this value will be clipped
val_slices: number of expected validations in one epoch
lr_decay_patience_steps: number of patience validation slices after which the learning rate scheduler will decay the learning rate
lr_decay_factor: the learning rate decay factor which is being used by the learning rate scheduler
lr_decay_threshold: the improvement threshold in validation bleu score below which will be a decay indicator signal to the learning rate scheduler
lr_decay_min: min value of learning rate which the decayed learning rate might approach 
```

# Experiment Results
In this section, we put the experiment results of different models on different datasets. Please check this page regularly as we add our new results below the previously posted ones.
 If you are in Natlang Lab or have access to internal SFU servers, you can pick the pre-trained models the result of which is posted here from `/cs/natlang-expts/hassan/SFUTranslate/pretrained/`.
 Each experiment will have a model file (ending in ".pt") in there with the exact same name mentioned in the table below.
 The dataset with which the model has been trained is put in a folder besides the model with the exact same name as the model.
  The configuration file with which the model was configured, can be downloaded by clicking on the experiment name link (first column of the table).

|                                      Experiment Name                                      	|                                      Replication Script                                      	|    Model    	|     Task     	|   Dataset   	|   Devset/Testset |    Language    	| Bleu Score (dev/test)	|                      More Info                      	|
|:-----------------------------------------------------------------------------------------:	|:-----------------------------------------------------------------------------------------:	|:-----------:	|:------------:	|:-----------:	|:-----------: |:--------------:	|:--------------------------:	|:---------------------------------------------------:	|
|         [seq2seq_multi30k_de_en](resources/nmt.yml)         	|         [replicate.sh](resources/exp-scripts/seq2seq_multi30k_de_en.sh)         	|   Seq2Seq   	|  Translation 	|  Multi30k2016 |           multi30k/val; multi30k/test2016   	| [German2English](http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/training.tar.gz) 	|           31.218 / 31.902           	| lowercased - tokenized with SpaCy	|
|         [seq2seq_iwslt_de_en](resources/exp-configs/seq2seq_de_en.yml)         	|         [replicate.sh](resources/exp-scripts/seq2seq_iwslt_de_en.sh)         	|   Seq2Seq   	|  Translation 	|  IWSLT2017 |           dev2010; tst201\[0-5\]   	| [German2English](https://wit3.fbk.eu/archive/2017-01-trnted/texts/de/en/de-en.tgz) 	|           26.153 / \[26.099; 28.383; 25.046; 27.021; 23.287; 23.277\]          	| lowercased - tokenized with SpaCy	|
|         [seq2seq_wmt_de_en](resources/exp-configs/seq2seq_wmt14_de_en.yml)         	|         [replicate.sh](resources/exp-scripts/seq2seq_wmt_de_en.sh)         	|   Seq2Seq   	|  Translation 	|  WMT14 |           newstest2009; newstest2016   	| [German2English](https://drive.google.com/uc?export=download&id=0B_bZck-ksdkpM25jRUN2X2UxMm8) 	|           17.941 / 22.603          	| lowercased - bpe tokenized \[32000 tokens\]|

## How to replicate our results?
Second column of the provided results table contains the replication scripts for each experiment 
(in cases of multiple test sets, the script only tests on the default test set provided by `torchtext`).
Replication of our results is quite easy, just download the `replicate.sh` script for the experiment you are interested in and run `bash <script_name>`.
The script will create a virtual environment, install SFUTranslate along with all of its dependencies, trains the model with our provided configuration, and test and print the dev/test scores. 
You can also modify these scripts as you desire. 

The validation results during the training as well as the samples created from the model in each epoch, can be found in `train.output` during and after training (it might take a bit till they actually appear in the file while training is in process as python buffers the content and writes them in batches).
The final validation and test results can be found in `test.output`. You may also track the training progress by running the command `cd /path/to/SFUTranslate/translate && tail -f train_progress_bars.log`.
The output files of our experiment are also uploaded besides each replication script in a folder with the same name, so you can compare and verify your replicated results.  

# Requirements and Dependencies
In this section, we go over the required libraries and how the project is dependant on each so that in case of the need 
to change (or remove) any of them, you know how and where to look for them.
The descriptions here are essentially describing the content of `requirements.txt` file besides this Readme document.

- [`pyyaml`](https://pyyaml.org/) the library needed to read the configurations, parse them and access their parsed values.
You may need to look at the content of `translate.configuration` class for the use case of this library.

- [`pytorch`](https://pytorch.org/docs/stable/index.html) the backend library which provides the neural network related 
functionality and classes. 

- [`torchtext`](https://torchtext.readthedocs.io/en/latest/) the data provider which downloads and loads the demanded dataset.

- [`sacrebleu`](https://github.com/mjpost/sacreBLEU) the evaluation package used for computing the Bleu scores in validation and test set score computation.

- [`sacremoses`](https://github.com/alvations/sacremoses) the tokenizer/detokenizer implementation of Moses.

- [`spaCy`](https://spacy.io/) the pre-processing toolkit used for normalization and tokenization of `English`, `German`
, `Spanish`, `Portuguese`, `French`, `Italian`, and `Dutch`. However, to make the library able to process each of the 
languages you will need to download its resources for spaCy using the following lines (you should simply copy the 
download line and past it into the command line to get executed).
```commandline
python -m spacy download en
python -m spacy download de
python -m spacy download es
python -m spacy download pt
python -m spacy download fr
python -m spacy download it
python -m spacy download nl
```

- [`subword_nmt`](https://github.com/rsennrich/subword-nmt) the implementation of the bye-pair encoding from the paper
[Neural Machine Translation of Rare Words with Subword Units](http://www.aclweb.org/anthology/P16-1162) used for providing the Byte-Pair level granularity.

- Utility libraries \[`tqdm` and `xml`\] the libraries that provide simple utility functionality.

 
# Help and Comments
If you need help regarding the toolkit or you want to discuss your comments, you are more than welcome to email [Hassan S.Shavarani](sshavara@sfu.ca).