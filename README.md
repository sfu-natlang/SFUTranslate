<img src="docs/logo.jpg" align="left"/>
<img src="docs/natlang-logo.png" align="right" width="150"/>
<h1 align="center"> SFUTranslate </h1>
<br/><br/>

This is an academic machine translation toolkit, in which the focus has been towards readability and changeability.
We also have tried to make the algorithms as fast as possible, but please let us know if you have any suggestions or
concerns regarding the toolkit. To get familiar with what you can do and how you can do it please read through this documentation.

To run the code you will need python 3.8+ (tested on v3.8.2) and PyTorch 1.6+.

# Getting Started

To get started, we start with the project structure. In the highest level of the project, there are three main directories:

- `resources`: All the necessary resources for the project are expected to be loaded from the `SFUTranslate/resources` directory.
Please refrain from putting the resources anywhere else if you are planing to make a pull request.
 
- `translate`: All the source code files are under `SFUTranslate/translate` directory.
If you are using an IDE to debug or run the code, don't forget to mark this directory as your sources directory.
Otherwise, you will need to have the address of `SFUTranslate/translate` directory in your `$PATH` environment variable to be able to run the code.
Another way of running the code would be to run `cd /path/to/SFUTranslate/translate && python trainer.py <path/to/config.yml>`. 
To test the trained model on the test set(s), you may use the `test_trained_model` script 
(e.g. `cd /path/to/SFUTranslate/translate && python test_trained_model.py <path/to/config.yml>`).
The last way of using the toolkit is to run the standalone experiment scripts in `resources/exp-scripts` directory. 
Please note that the scripts will start by downloading and setting up an independent python environment, and they don't need your interaction while running.

- `tests`: All the source code for unit test cases for different modules of the toolkit.
If you are using an IDE to debug or run the code, don't forget to mark this directory as your test sources directory.
 
The next sections will help you get more familiar with the code flow and training different models.

## Code Structure
As stated earlier, the source code files are in the `translate` directory (i.e. `/path/to/SFUTranslate/translate/`). 

The following figure depicts the general structure of the modules.

<img src="docs/SFUTranslate.svg?sanitize=True"/>

As of the current version, the `translate` package contains the following sub-packages and classes.
  - `models`
    + `general`  
    + `aspects` the package containing implementations of aspect-augmented nmt model and its baselines  
        - `ae_utils`
        - `aspect_extract_main`
        - `containers`
        - `extract_vocab`
        - `model`
        - `module`
        - `tester`
        - `trainer`
    + `sts` the package containing the implementation of the attentional sequence-to-sequence model using RNNs
        - `model`
    + `copy` the package containing the implementation of a simple copy model which only tokenizes and then detokenizes the input to produce the output.
        - `model`
    + `transformer` the package containing the implementation of vanilla transformer model
        - `model`
        - `modules`
        - `optim`
        - `utils`
  - `readers` the data read/preprocess methods and classes are placed in this package
    + `datasets`
        - `generic`
        - `dataset`
    + `data_provider`   
    + `iterators`
    + `sequence_alignment`
    + `tokenizers`
    + `utils`
  - `scripts` the package containing all the dangling scripts in the toolkit
    + `create_pretrained_tokenizer_vocabulary`
    + `extract_common_vocab`
  - `utils` the package containing the utility functions used in the toolkit 
    + `containers`   
    + `evaluation`
    + `init_nn`
    + `optimizers`
  - `configuration` the class in charge of reading the config file and providing its content inside an easily accessible configuration object (`cfg`).
  - `test_trained_model` the test script which loads an already trained model (pointed from the config file) and runs the beam search on it.
  - `trainer` the main script which loads the config file, creates the model and runs the training process. 
  You may want to start looking into this script first, to get familiar with what can be done using this toolkit.
  

## What can be put in the config file?
Here we present a complete schema for the config file containing all possible valid tags that can be put in the config file.
Please note that you may put some in and remove some from your config file, however, if the config file lacks the configurations 
that are essential to your task you will face an error indicating that the required configuration value is not present.
  In that case, please look at the config file and put the configuration tag with your desired value in it. An example 
  config file called `nmt.yml` is already put in the `resources` directory. You can modify and use it for running the 
  project. Nevertheless, you can create your own config file as a text file with a `.yml` extension in the name and put your 
  configurations in it. Here is the configuration schema:
```yamlex
debug_mode: [true/false] if true tokenizer is deactivated and length filter is also applied to validation and test sets
src_lang: the bi-letter language identifier for source langugage
tgt_lang: the bi-letter language identifier for target langugage
dataset_name: the name of the torchtext datasetname [currently supported: "multi30k16", "iwslt17", "wmt19_de_en", "wmt19_de_fr"]
lowercase_data: [true/false] whether the dataset setences need to be lowercased or not
src_tokenizer: the tokenizer to be used for the source side of parallel data [possible values: "generic"|"moses"|"pre_trained"|"spacy"|"bert"] 
tgt_tokenizer: the tokenizer to be used for the target side of parallel data [possible values: "generic"|"moses"|"pre_trained"|"spacy"|"bert"]
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
extract_unk_stats: the flag which enables the code to perform type/token analysis on the ratio of <UNK> tokens in the current vocabulary settings
share_vocabulary: the flag which enables merging the source and target vocabulary into a single object, assigning unique ids to the same token in both source and target space
sentence_count_limit: the maximum number of sentences to be considered from the trainset. it will normally be used when different data fractions are intended to be compared. please note that the actual number of processed sentences can be lower than this number since empty lines are removed from training data. 
aspect_vectors_data_address: the address of the pre-trained aspect vector extractors. You don't need to set any value for it if your model is not "aspect_augmented_transformer". 

model_name: `sts` or `transformer` [also "aspect_augmented_transformer", "multi_head_aspect_augmented_transformer", "syntax_infused_transformer", and "bert_freeze_input_transformer"]
train_batch_size: the average number of words expected to be put in a batch while training [4000 to 5000 seem to be a reasonable defalt]
valid_batch_size: the average number of words expected to be put in a batch while testing [you dont need big numbers in this case]
maximum_decoding_length: maximum valid decoding length

emb_dropout: the droupout ratio applied to both source and target embedding layers of sts model
encoder_emb_size: the size of the embedding layer in encoder in sts model
encoder_hidden_size: the size of the RNN layer in encoder in sts model
encoder_layers: number of encoder RNN layers in sts model
encoder_dropout_rate: the dropout value applied to the encoder RNN in sts model
decoder_emb_size: the size of the embedding layer in decoder in sts model
decoder_hidden_size: the size of the RNN layer in decoder in sts model if this is not equal to twice the size of encoder output, a bridge network automatically mapps the encoder hidden states to the decoder hidden states.
decoder_layers: number of decoder RNN layers in sts model
decoder_dropout_rate: the dropout value applied to the decoder RNN in sts model
out_dropout: the dropout value applied to the output of attention before getting fed to the affine transformation in sts model
coverage_dropout: the droput value applied to the encoded representations in sts model before phi parameter is created

bahdanau_attention: [true/false] if not true Loung general attention is applied (in sts model)
coverage_phi_n: the N value for calculating the phi parameter in linguistic coverage calculation (in sts model)
coverage_required: [true/false] indicates whether coverage mechanism will be conisdered or not (in sts model)
coverage_lambda: the justification coefficient for the coverage loss when its added to the NLL loss (in sts model)

transformer_d_model: the size of the encoder and decoder layers of the transformer model
transformer_h: the number of the transformer model heads
transformer_dropout: the droput applied to the transformer model layers
transformer_d_ff: the size of the feed-forward layer in the transformer model
transformer_max_len: max expected length of input for positioanl encoding in tranformer model [you can safely leave it be as 5000 if you dont do document translation]
transformer_N: the number of encoder and decoder layers in the transformer model
transformer_loss_smoothing: the soothing factor in KL divergance loss calculation
transformer_opt_factor: the NoamOpt scheduler learning rate multiplicatin factor
transformer_opt_warmup: the number of NoamOpt scheduler warmup steps
share_all_embeddings: the flag indicating the embeddings are required to be shared in encoder/decoder modules

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
update_freq: a non-negative integer indicating the number of batches, the gradient of which are accumulated before calling optimizer.step() [for more details see https://arxiv.org/pdf/1806.00187.pdf]

beam_size: the size of each bucket considered in beam search
beam_search_length_norm_factor: the length normalization factor in beam search based on Google's NMT system paper [https://arxiv.org/pdf/1609.08144.pdf]
beam_search_coverage_penalty_factor: the coverage penalty factor in beam search based on Google's NMT system paper [https://arxiv.org/pdf/1609.08144.pdf] 
checkpoint_name: the name of the checkpoint which is being saved/loaded  
```

# Aspect Augmented NMT Experiment Results
In this section, we report the results of our Aspect Integrated NMT results along with the replicated baseline results. 
You can run our experiments using the standalone scripts in `resources/exp-scripts/aspect_exps`. 
The source code for Aspect Augmented NMT along with the replicated baselines is implemented in `translate/models/aspects`.
In the results tables:

- \#param represents the number of trainable parameters (size of BERT model parameters \[110.5M\] has not been added to the model size for the aspect augmented and bert-freeze models since BERT is not trained in these settings).
- runtime is the total time the training script has ran and includes time taken for creating the model, reading the data and iterating over the instances for all the epochs.
- We have used a single GeForce GTX 1080 GPU for M30k experiments and a single Titan RTX GPU for IWSLT and WMT experiments.

## M30k German to English
|                                        | val    | test2016 | \#param | runtime |
|----------------------------------------|--------|----------|---------|---------|
| Vaswani et al\. 2017                   | 38\.00 | 37\.25   | 9\.5 M  | 84 min  |
| Sundararaman et al\. 2019              | 38\.96 | 36\.82   | 13\.9 M | 514 min |
| Clinchant et al\. 2019 \(bert freeze\) | 38\.76 | 37\.72   | 9\.1 M  | 99 min  |
| Aspect Augmented \+M30k asp\. vectors  | **39\.82** | 38\.36   | 10\.1 M | 104 min |
| Aspect Augmented \+WMT asp\. vectors   | 38\.97 | **39\.28**   | 10\.1 M | 102 min |

## M30k German to French
|                                        | val    | test2016 | \#param | runtime |
|----------------------------------------|--------|----------|---------|---------|
| Vaswani et al\. 2017                   | 31\.01 | 30\.27   | 9\.4 M  | 93 min  |
| Sundararaman et al\. 2019              | 33\.02 | 32\.99   | 13\.6 M | 504 min |
| Clinchant et al\. 2019 \(bert freeze\) | 33\.71 | 32\.85   | 9\.0 M  | 104 min |
| Aspect Augmented \+M30k asp\. vectors  | 34\.11 | 33\.90   | 9\.9 M  | 108 min |
| Aspect Augmented \+WMT asp\. vectors   | **34\.90** | **33\.94**   | 9\.9 M  | 118 min |

## IWSLT17 German to English
|                                        | dev2010        | tst2010        | tst2011        | tst2012        | tst2013        | tst2014        | tst2015        | \#param | runtime  |
|----------------------------------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|---------|----------|
| Vaswani et al\. 2017                   | 29\.57         | 29\.72         | 31\.82         | 28\.93         | 30\.94         | 26\.21         | 26\.80         | 18\.4 M | 172 min  |
| Sundararaman et al\. 2019              | 30\.93         | 31\.49         | 32\.82         | 29\.64         | 31\.79         | 27\.51         | 27\.47         | 28\.9 M | 1418 min |
| Clinchant et al\. 2019 \(bert freeze\) | 30\.79         | 31\.03         | 33\.30         | 30\.00         | 31\.50         | 27\.12         | 26\.97         | 18\.0 M | 212 min  |
| Aspect Augmented \+IWSLT asp\. vectors | 30\.54         | 31\.18         | 33\.87         | 30\.09         | 31\.58         | 27\.94         | 28\.15         | 18\.9 M | 214 min  |
| Aspect Augmented \+WMT asp\. vectors   | **32\.60**     | **32\.77**     | **34\.73**     | **30\.71**     | **32\.71**     | **28\.19**     | **28\.28**     | 18\.9 M | 211 min  |

## WMT14 German to English
|                                        | wmt\_val        | newstest2014   | newstest2015   | newstest2016   | newstest2017   | newstest2018   | newstest2019   | \#param | runtime |
|----------------------------------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|---------|---------|
| Vaswani et al\. 2017                   | 28\.94         | 27\.19         | 27\.45         | 31\.90         | 28\.09         | 33\.97         | 30\.43         | 68\.7 M | 35 h    |
| Sundararaman et al\. 2019              | 29\.12         | 27\.33         | 27\.35         | **32\.49**     | 28\.36         | 34\.72         | 31\.12         | 93\.8 M | 258 h   |
| Clinchant et al\. 2019 \(bert freeze\) | 28\.65         | 27\.14         | 27\.35         | 31\.15         | 27\.75         | 34\.07         | 31\.04         | 69\.1 M | 33 h    |
| Aspect Augmented \+WMT asp\. vectors   | **29\.16**     | **28\.01**     | **28\.42**     | 32\.04         | **28\.85**     | **35\.35**     | **31\.83**     | 70\.3 M | 46 h    |


# Older Experiment Results
In this section, we put the experiment results of different models on different datasets. 
Please check this page regularly as we add our new results below the previously posted ones.
You can pick the pre-trained models the result of which are posted here from the hyperlinks of the `Pretrained Models`. 
To use them you can simply put them in a directory named `.checkpoints` created besides the `translate` package (if it is a zipped file, unzip it there) and point the `checkpoint_name` in the configuration file to the downloaded pre-trained model. 
Each experiment will have a model file (ending in ".pt") in there with the exact same name mentioned in the table below.
The dataset with which the model has been trained will be automatically downloaded.
The configuration file with which the model was configured, can be downloaded by clicking on the experiment name link (first column of the table).

|                                      Experiment Name                                      	|                                      Replication Script                                      	|    Model    	|     Task     	|   Dataset   	|   Devset/Testset |    Language    	| Bleu Score (dev/test)	| Model File         |                      More Info                      	|
|:-----------------------------------------------------------------------------------------:	|:-----------------------------------------------------------------------------------------:	|:-----------:	|:------------:	|:-----------:	|:-----------: |:--------------:	|:--------------------------: |:------------------:	|:---------------------------------------------------:	|
|         [seq2seq_multi30k_de_en](resources/exp-configs/seq2seq_multi30k_de_en.yml)         	|         [replicate.sh](resources/exp-scripts/seq2seq_multi30k_de_en.sh)         	|   Seq2Seq   	|  Translation 	|  Multi30k2016 |           multi30k/val; multi30k/test2016   	| [German2English](http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/training.tar.gz) 	|           33.504 / 33.840           | [Pretrained Model](https://drive.google.com/open?id=16UOg06bq4swoOEQ_xBMStZJDjtXSAgCN)	| lowercased - tokenized with SpaCy - &#124;V&#124;=3000	|
|         [transformer_multi30k_de_en](resources/exp-configs/transformer_multi30k_de_en.yml)         	|         [replicate.sh](resources/exp-scripts/transformer_multi30k_de_en.sh)         	|   Transformer   	|  Translation 	|  Multi30k2016 |           multi30k/val; multi30k/test2016   	| [German2English](http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/training.tar.gz) 	|           34.652 / 34.727          | [Pretrained Model](https://drive.google.com/open?id=1XzSRtQzLwLADNtWsR3nC1uldUjLO0YL0)	| lowercased - tokenized with SpaCy - &#124;V&#124;=3000	|
|         [seq2seq_iwslt_de_en](resources/exp-configs/seq2seq_iwslt_de_en.yml)         	|         [replicate.sh](resources/exp-scripts/seq2seq_iwslt_de_en.sh)         	|   Seq2Seq   	|  Translation 	|  IWSLT2017 |           dev2010; tst201\[0-5\]   	| [German2English](https://wit3.fbk.eu/archive/2017-01-trnted/texts/de/en/de-en.tgz) 	|           26.438 / \[26.226; 28.378; 25.208; 27.523; 23.219; 23.604\]         | [Pretrained Model](https://drive.google.com/open?id=1GquqMA_EJvdQLisT4-hByv1SygVeQQ9Z) 	| lowercased - tokenized with SpaCy	|
|         [transformer_iwslt_de_en](resources/exp-configs/transformer_iwslt_de_en.yml)         	|         [replicate.sh](resources/exp-scripts/transformer_iwslt_de_en.sh)         	|   Transformer   	|  Translation 	|  IWSLT2017 |           dev2010; tst201\[0-5\]   	| [German2English](https://wit3.fbk.eu/archive/2017-01-trnted/texts/de/en/de-en.tgz) 	|           27.515 / \[28.184; 30.316; 26.905; 29.206; 24.313; 25.267\]       | [Pretrained Model](https://drive.google.com/open?id=1TB9PrlnqgtOodx4C0H2BQuk4nk7ZNd0_)   	| lowercased - tokenized with SpaCy	|
|         [seq2seq_wmt_de_en](resources/exp-configs/seq2seq_wmt19_de_en.yml)         	|         [replicate.sh](resources/exp-scripts/seq2seq_wmt_de_en.sh)         	|   Seq2Seq   	|  Translation 	|  WMT19 |           valid (1% of train data); newstest201\[4-9\]   	| [German2English](https://drive.google.com/uc?export=download&id=0B_bZck-ksdkpM25jRUN2X2UxMm8) 	|           28.487 / \[21.046; 21.767; 25.274; 22.337; 26.566; 24.261\]          | [Pretrained Model](https://drive.google.com/open?id=1-nTRXGNrLdBATRvXdfRNz5h7f5OP6f2S)	| lowercased - bpe tokenized \[40000 tokens\]|
|         [transformer_wmt_de_en](resources/exp-configs/transformer_wmt19_de_en.yml)         	|         [replicate.sh](resources/exp-scripts/transformer_wmt_de_en.sh)         	|   Transformer   	|  Translation 	|  WMT19 |           valid (1% of train data); newstest201\[4-9\]   	| [German2English](https://drive.google.com/uc?export=download&id=0B_bZck-ksdkpM25jRUN2X2UxMm8) 	|           27.208 / \[20.408; 21.034; 24.078; 21.317; 25.098; 23.304\]          | [Pretrained Model](https://drive.google.com/open?id=1dSrib2mF7k2LxZLcmE74RlLPqVB9E8lo)	| lowercased - bpe tokenized \[40000 tokens\]|

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
You don't have to use spacy as `readers.tokenizers` providers a number of different tokenizers you can choose among.
```commandline
python -m spacy download en
python -m spacy download de
python -m spacy download es
python -m spacy download pt
python -m spacy download fr
python -m spacy download it
python -m spacy download nl
```

- [`transformers`](https://github.com/huggingface/transformers) the library providing pre-trained bert and bert-tokenizer models. 

- [`unidecode`](https://github.com/avian2/unidecode) the library used for transliteration of unicode text into ascii (in `readers.sequence-alignment`) script.

- [`tokenizers`](https://github.com/huggingface/tokenizers) the tokenizer library providing state-of-the-art word-piece tokenization and pre-training models for both sub-word and word-piece.

- [`textblob`](https://textblob.readthedocs.io/en/dev/) the library used for sentiment analysis feature extraction in `models.aspects.extract_vocab`.

- [`nltk`](https://www.nltk.org/) the provider of lesk algorithm implementation for word-sense feature extraction in `models.aspects.extract_vocab`.

- [`scikit-learn`](https://scikit-learn.org/stable/index.html) the classification metrics/report provider library for aspect extractor

- [`numpy`](https://numpy.org/) the library used to create temporary tensors on cpu before copy to torch and gpu.

- Utility libraries \[`tqdm` and `xml`\] the libraries that provide simple utility functionality.

 
# Help and Comments
If you need help regarding the toolkit or you want to discuss your comments, you are more than welcome to email [Hassan S.Shavarani](sshavara@sfu.ca).