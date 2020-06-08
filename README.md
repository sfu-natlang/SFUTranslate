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
        - `model`
    + `transformer`
        - `model`
        - `modules`
        - `optim`
        - `utils`
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
debug_mode: [true/false] if true tokenizer is deactivated and length filter is also applied to validation and test sets
src_lang: the bi-letter language identifier for source langugage
tgt_lang: the bi-letter language identifier for target langugage
dataset_name: the name of the torchtext datasetname
lowercase_data: [true/false] whether the dataset setences need to be lowercased or not
src_tokenizer: the tokenizer to be used for the source side of parallel data [possible values: "generic"|"moses"|"pre_trained"] 
tgt_tokenizer: the tokenizer to be used for the target side of parallel data [possible values: "generic"|"moses"|"pre_trained"]
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

model_name: `transformer` or `sts`
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

# Experiment Results
In this section, we put the experiment results of different models on different datasets. Please check this page regularly as we add our new results below the previously posted ones.
 You can pick the pre-trained models the result of which are posted here from the hyperlinks of the `Pretrained Models`. To use them you can simply put them besides the `trainer.py` script (if it is a zipped file, unzip it there) and point the `checkpoint_name` in the configuration file to the downloaded pre-trained model. 
 Each experiment will have a model file (ending in ".pt") in there with the exact same name mentioned in the table below.
 The dataset with which the model has been trained is put in a folder besides the model with the exact same name as the model.
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