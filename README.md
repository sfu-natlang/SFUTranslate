<img src="docs/logo.jpg" align="left"/>
<img src="docs/natlang-logo.png" align="right" width="150"/>
<h1 align="center"> SFUTranslate </h1>
<br/><br/>

This is an academic machine translation toolkit, in which the focus has been towards readability and changeability.
We also have tried to make the algorithms as fast as possible, but please let us know if you have any suggestions or
concerns regarding the toolkit. To get familiar with what you can do and how you can do it please read through this documentation.

To run the code you will need python 3.8+ (tested on v3.8.2) and PyTorch 1.6+.

**This is the library that contains the implementation for our EACL 2021 paper "Better Neural Machine Translation by Extracting Linguistic Information from BERT"**. 
For updated aspect-augmented translation results and acquiring aspect vectors please look down this document. 

# Getting Started

We start with the project structure. In the highest level of the project, there are four main directories:

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
  
- `docs`: Containing documentation related files.
 
The next sections will help you get more familiar with the code flow and training different models.

## Code Structure
As stated earlier, the source code files are in the `translate` directory (i.e. `/path/to/SFUTranslate/translate/`). 

The following figure depicts the general structure of the modules.

<img src="docs/SFUTranslate.svg?sanitize=True"/>

As of the current version, the `translate` package contains the following sub-packages and classes.
  - `models`
    + `general` providing a general wireframe for a new `NMTModel` instance.
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
        - `torch_model`
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
    + `mteval-v14.pl` (its required dependencies are placed besides it in `Sort` directory)
    + `wrap-xml.perl`
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
Here we present a complete schema for the config file containing all possible valid tags that can be put in it.
Please note that you may put some in and remove some from your config file, however, if it lacks the configurations 
that are essential to your task you will face an error indicating that the required configuration value is not present.
In that case, please look at the config file and put the configuration tag with your desired value in it. 
An example config file called `nmt.yml` is already put in the `resources` directory. You can modify and use it for running the project. 
Nevertheless, you can create your own config file as a text file with a `.yml` extension in the name and put your configurations in it. 
Here is the configuration schema:
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
In this section, we report the results of our Aspect Integrated NMT model along with the replicated baseline models. 
The results here are the same as what was reported in the paper. 
You can run our experiments using the standalone scripts in `resources/exp-scripts/aspect_exps`. 
The source code for Aspect Augmented NMT along with the replicated baselines is implemented in `translate/models/aspects`.
In the result tables:

- \#param represents the number of trainable parameters (size of BERT model parameters \[110.5M\] has not been added to the model size for the aspect augmented and bert-freeze models since BERT is not trained in these settings).
- runtime is the total time the training script has ran and includes time taken for creating the model, reading the data and iterating over the instances for all the epochs.
- We have used a single GeForce GTX 1080 GPU for M30k experiments and a single Titan RTX GPU for IWSLT and WMT experiments.
- The following results are created by lower-casing, translation, and true-casing afterwards. 
  The cased aspect-augmneted experiment results are also available and although we have not reported them in the paper, you can find them in the last table.
- All the experiment results are calculated using `mteval-v14.pl` script.
## M30k German to English
|                                        | val    | test2016 | \#param | runtime |
|----------------------------------------|--------|----------|---------|---------|
| Vaswani et al\. 2017                   | 39\.63 | 38\.25   | 9\.5 M  | 84 min  |
| Sundararaman et al\. 2019              | 40\.03 | 38\.32   | 13\.9 M | 514 min |
| Clinchant et al\. 2019 \(bert freeze\) | 40\.07 | 39\.73   | 9\.1 M  | 99 min  |
| Aspect Augmented \+M30k asp\. vectors  | **40\.47** | 40\.19   | 10\.1 M | 104 min |
| Aspect Augmented \+WMT asp\. vectors   | 38\.72 | **41\.53**   | 10\.1 M | 102 min |

## M30k German to French
|                                        | val    | test2016 | \#param | runtime |
|----------------------------------------|--------|----------|---------|---------|
| Vaswani et al\. 2017                   | 31\.07 | 30\.29   | 9\.4 M  | 93 min  |
| Sundararaman et al\. 2019              | 32\.55 | 32\.71   | 13\.6 M | 504 min |
| Clinchant et al\. 2019 \(bert freeze\) | 33\.83 | 33\.15   | 9\.0 M  | 104 min |
| Aspect Augmented \+M30k asp\. vectors  | 34\.45 | **34\.42**   | 9\.9 M  | 108 min |
| Aspect Augmented \+WMT asp\. vectors   | **34\.73** | 34\.28   | 9\.9 M  | 118 min |

## IWSLT17 German to English
|                                        | dev2010        | tst2010        | tst2011        | tst2012        | tst2013        | tst2014        | tst2015        | \#param | runtime  |
|----------------------------------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|---------|----------|
| Vaswani et al\. 2017                   | 27\.69         | 27\.93         | 31\.88         | 28\.15         | 29\.59         | 25\.66         | 26\.76         | 18\.4 M | 172 min  |
| Sundararaman et al\. 2019              | 29\.53         | 29\.67         | 33\.11         | 29\.42         | 30\.89         | 27\.09         | 27\.78         | 28\.9 M | 1418 min |
| Clinchant et al\. 2019 \(bert freeze\) | 30\.31         | 30\.00         | 34\.20         | 30\.04         | 31\.26         | 27\.50         | 27\.88         | 18\.0 M | 212 min  |
| Aspect Augmented \+IWSLT asp\. vectors | 29\.03         | 29\.17         | 33\.42         | 29\.58         | 30\.63         | 26\.86         | 27\.83         | 18\.9 M | 214 min  |
| Aspect Augmented \+WMT asp\. vectors   | **31\.22**     | **30\.82**     | **34\.79**     | **30\.29**     | **32\.34**     | **27\.71**     | **28\.40**     | 18\.9 M | 211 min  |

## WMT14 German to English
|                                        | wmt\_val        | newstest2014   | newstest2015   | newstest2016   | newstest2017   | newstest2018   | newstest2019   | \#param | runtime |
|----------------------------------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|---------|---------|
| Vaswani et al\. 2017                   | 28\.96         | 26\.91         | 26\.91         | 31\.42         | 28\.07         | 33\.56         | 29\.77         | 68\.7 M | 35 h    |
| Sundararaman et al\. 2019              | 28\.56         | 27\.80         | 26\.93         | 30\.44         | 28\.63         | 33\.87         | 30\.48         | 93\.8 M | 258 h   |
| Clinchant et al\. 2019 \(bert freeze\) | 28\.63         | 27\.54         | 27\.15         | 31\.69         | 28\.30         | 33\.89         | **31\.48**     | 69\.1 M | 33 h    |
| Aspect Augmented \+WMT asp\. vectors   | **28\.98**     | **28\.05**     | **27\.58**     | **32\.29**     | **29\.07**     | **34\.74**     | **31\.48**     | 70\.3 M | 46 h    |

# \[Updated\] Cased Aspect Augmented NMT Experiment Results
As we suggested earlier, we have also trained aspect-augmented translation models using cased data, however, the results were not ready in time to put in the paper. 
This section reports the cased aspect-augmented translation results. 

## Configuration and loading pre-trained aspect extractor modules
To put more emphasis, we have used the tag `lowercase_data: false` in the config files when training aspect vectors and 
translation models and this is now the default in all configuration files under `exp-configs/aspect_exps`.   
We are also releasing only this set of German aspect vectors (trained with cased data) to be used in other tasks.
To replicate our experiments, please click on the dataset name on the first column of each table and download the linked aspect extractor module
(the pre-trained aspect extractor module will be used in `models.aspects.model.AspectAugmentedTransformer` class and 
you can take a look at it for better understanding).
Once downloaded, put the pre-trained aspect extractor module in `.checkpoints` directory (if it doesn't exist create it) besides the `translate` and `resources` directories.
Now point the `aspect_vectors_data_address` tag in the config file of your desired experiment to the aspect extractor module
(e.g. `aspect_vectors_data_address: ../.checkpoints/aspect_extractor.de`) and run the trainer with the modified config file.

## Updated results
We retrain aspect extractors using cased data. First, lets look at the cased classification f-1 scores on the same aspect tag set (in word-level).

| Dataset | CPOS | FPOS | WSH  | #Tokens |
|---------|------|------|------|---------|
| M30k    |98\.87|98\.21|99\.34|12822    |
| IWSLT   |97\.79|96\.68|98\.71|16760    |
| WMT     |94\.59|94\.28|88\.74|50666    |

Also, lets look at the sub-word level classification f-1 scores.

| Dataset | CPOS | FPOS | WSH  | SWP  | #Tokens |
|---------|------|------|------|------|---------|
| M30k    |98\.13|97\.34|99\.44|99\.86|15836    |
| IWSLT   |96\.78|95\.47|98\.87|99\.70|19559    |
| WMT     |91\.96|91\.49|89\.03|97\.70|63259    |

Now, we report both cased Bleu and NIST scores from `mteval-v14.pl` for the cased aspect-augmented NMT experiment.

| German to English           | val            | test_2016_flickr | test_2017_flickr | test_2018_flickr | test_2017_mscoco |
|-----------------------------|----------------|------------------|------------------|------------------|------------------|
| [Using M30k aspect vectors](resources/aspect-extractors/cased-m30k-aspect-extractors.de)   | 41\.82 / 7\.88 | 41\.48 / 7\.98   | 38\.07 / 7\.33   | 33\.40 / 6\.97   | 30\.50 / 6\.21   |

| German to English           | dev2010        | tst2010        | tst2011        | tst2012        | tst2013        | tst2014        | tst2015        |
|-----------------------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|
| [Using IWSLT aspect vectors](resources/aspect-extractors/cased-iwslt-aspect-extractors.de)  | 32\.84 / 7\.53   | 32\.14 / 7\.45 | 35\.88 / 7.87  | 31\.33 / 7\.41 | 33\.83 / 7\.51 | 29\.14 / 7\.01 | 30\.30 / 7\.04 |


| German to English           | wmt\_val       | newstest2014   | newstest2015   | newstest2016   | newstest2017   | newstest2018   | newstest2019   | newstest2020 | newstestB2020 |
|-----------------------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|--------------|---------------|
| [Using WMT aspect vectors](resources/aspect-extractors/cased-wmt-aspect-extractors.de)    |29\.74 / 7\.19  |29\.58 / 7.69   |29\.56 / 7\.55  |34\.38 / 8\.33  | 30\.83 / 7\.84 |36\.75 / 8\.73  |31\.51 / 7\.62  |18\.70 / 4\.14| 18\.21 / 4\.17|

# Older Experiment Configurations
In this section, we put the experiment configurations of different models on different datasets.
You can pick the pre-trained models the result of which are posted here from the hyperlinks of the `Pretrained Models` (**the models are currently removed as they are pretty outdated!**). 
To use them you can simply put them in a directory named `.checkpoints` created besides the `translate` package (if it is a zipped file, unzip it there) and point the `checkpoint_name` in the configuration file to the downloaded pre-trained model. 
Each experiment will have a model file (ending in ".pt") in there with the exact same name mentioned in the table below.
The dataset with which the model has been trained will be automatically downloaded.
The configuration file with which the model was configured, can be downloaded by clicking on the experiment name link (first column of the table).

|                                      Experiment Name                                      	|                                      Replication Script                      	|    Model    	|     Task     	|   Dataset   	|   Devset/Testset                               |    Language    	                                                                                | Bleu Score (dev/test)	| Model File |   More Info  |
|:-----------------------------------------------------------------------------------------:	|:----------------------------------------------------------------------------:	|:-----------:	|:------------:	|:-----------:	|:---------------------------------------------: |:-----------------------------------------------------------------------------------------------:	|:--------------------: |:---------: |:--------:	|
|         [seq2seq_multi30k_de_en](resources/exp-configs/seq2seq_multi30k_de_en.yml)         	|         [replicate.sh](resources/exp-scripts/seq2seq_multi30k_de_en.sh)       |   Seq2Seq   	|  Translation 	|  Multi30k2016 | multi30k/val; multi30k/test2016   	         | [German2English](http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/training.tar.gz) 	            |           -           |            |          -	|
|         [transformer_multi30k_de_en](resources/exp-configs/transformer_multi30k_de_en.yml)    |         [replicate.sh](resources/exp-scripts/transformer_multi30k_de_en.sh)   |   Transformer |  Translation 	|  Multi30k2016 | multi30k/val; multi30k/test2016                | [German2English](http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/training.tar.gz) 	            |           -           |            |          -	|
|         [seq2seq_iwslt_de_en](resources/exp-configs/seq2seq_iwslt_de_en.yml)         	        |         [replicate.sh](resources/exp-scripts/seq2seq_iwslt_de_en.sh)         	|   Seq2Seq   	|  Translation 	|  IWSLT2017    | dev2010; tst201\[0-5\]   	                     | [German2English](https://wit3.fbk.eu/archive/2017-01-trnted/texts/de/en/de-en.tgz) 	            |           -           |            |          -	|
|         [transformer_iwslt_de_en](resources/exp-configs/transformer_iwslt_de_en.yml)         	|         [replicate.sh](resources/exp-scripts/transformer_iwslt_de_en.sh)      |   Transformer |  Translation 	|  IWSLT2017    | dev2010; tst201\[0-5\]   	                     | [German2English](https://wit3.fbk.eu/archive/2017-01-trnted/texts/de/en/de-en.tgz) 	            |           -           |            |          -	|
|         [seq2seq_wmt_de_en](resources/exp-configs/seq2seq_wmt19_de_en.yml)         	        |         [replicate.sh](resources/exp-scripts/seq2seq_wmt_de_en.sh)         	|   Seq2Seq   	|  Translation 	|  WMT19        | valid (1% of train data); newstest201\[4-9\]   | [German2English](https://drive.google.com/uc?export=download&id=0B_bZck-ksdkpM25jRUN2X2UxMm8) 	|           -           |            |          -	|
|         [transformer_wmt_de_en](resources/exp-configs/transformer_wmt19_de_en.yml)         	|         [replicate.sh](resources/exp-scripts/transformer_wmt_de_en.sh)        |   Transformer |  Translation 	|  WMT19        | valid (1% of train data); newstest201\[4-9\]   | [German2English](https://drive.google.com/uc?export=download&id=0B_bZck-ksdkpM25jRUN2X2UxMm8) 	|           -           |            |          -   |

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