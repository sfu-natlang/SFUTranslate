<img src="resources/documents/logo.jpg" align="left"/>
<img src="resources/documents/natlang-logo.png" align="right" width="150"/>
<h1 align="center"> SFUTranslate </h1>
<br/><br/>

This is an academic machine translation toolkit, in which the main focus has been towards readability and changeability.
We also have tried to make the algorithms as fast as possible, but please let us know if you have any suggestions or
concerns regarding the toolkit. To get familiar with what you can do and how you can do it please read through this documentation.

To run the code you will need python 3.5+ and PyTorch 0.4+.

# Getting Started

To get started, we start with the project structure. In the highest level of the project, there are two main directories:

- `resurces`: All the necessary resources for the project are supposed to be loaded from the `SFUTranslate/resources` directory.
Please refrain from putting the resources anywhere else if you are planing to do a pull request.
 
- `src`: All the source code files are placed under `SFUTranslate/src` directory.
If you are using an IDE to debug or run the code, don't forget to mark this directory as your sources directory.
Otherwise you will need to have the address of `SFUTranslate/src` directory in your `$PATH` environment variable to be able to run the code.
Another way of running the code would be to run `cd /path/to/SFUTranslate/src && python -m package.subpackage.source_code_name <arguments>`.
As an example you can run the dummy Seq2Seq (or language model) trainer using the command  `cd /path/to/SFUTranslate/src && python -m translate.learning.trainer dummy.yaml`, considering that you have a YAML config file in the `/path/to/SFUTranslate/resources` directory.
 
The next sections will help you get more familiar with the code flow and training different models.

## Code Structure
As stated earlier, the source codes are placed in the `src` directory and more strictly every thing is placed in 
`translate` package (i.e. `/path/to/SFUTranslate/src/translate/`). 

The general structure of the modules is depicted in the following figure.

<img src="resources/documents/SFUTranslate.svg?sanitize=True"/>

As of the current version, the `translate` package contains the following packages.

- `backend` in charge of providing functionalities directly related to the backend (e.g. PyTorch framework). 
The package is designed the way that changing the backend from torch to any other framework (e.g. tensorflow)
would be fast and simple. 

- `configs` in charge of reading, parsing, and providing parsed configuration information to the project modules
using the config file passed to it (in YAML/Ansible format). You can look at `SFUTranslate/resources/dummy.yaml` to 
figure out the format of this configuration file.

- `learning` the package containing the modules and models defined in the project. In addition to the modules and models, three important parts of the project are located in this package.
  - `trainer` the main script which loads the config file, creates the model and runs the training process. 
  You may want to start looking into this script first, to get familiar with what can be done using this toolkit.
  Please note that with the current implementation of the `trainer` to add more dataset readers and more models, you will need to extend this script to support your readers/models.
  - `estimator` the script containing the `Estimator` class which will do forward and backward passes in to train the model and evaluate it.
  - `modelling` the abstract class which unifies the interface of models defined in the project. If you are making a new model (e.g. a sequence to sequence model or a language model) using the modules defined in the project (or the modules you have defined and added to the project), your model class needs to extend the `AbsCompleteModel` in this script.
- `logging` creates and provides a single instance of logger for all that needs to be logged across the project.

- `readers` provides all the funcionalities necessary for reading, understanding and processing the datasets. 
You may want to write your own dataset reader, in which case your dataset reader class must extend the abstact 
`AbsDatasetReader` class defined in this package. The dummy dataset providers `ReverseCopyDataset` and 
`SimpleGrammerLMDataset` are examples you can look at to understand how you may create your dataset provider.  

## What can be put in the config file?
Here we present a complete schema for the config file containing all possible valid tags that can be put in the config file.
Please note that you may put some in and remove some from your config file, however, if the config file lacks the configurations 
that are essential to your task you will face the error `The configuration value <CONFIG_VALUE> must exist in your configuration file!`.
  In that case, please look at the config file and put the configuration tag with your desired value in it. An example 
  config file called `dummy.yaml` is already put in the `resources` directory so you can modify and use for running the 
  project. Nevertheless, you can create your own config file as a text file whose name is ending in `.yaml` and put your 
  configurations in it. Here is the configuration schema:
```yamlex
reader:
    dataset:
        type: possible values [mono | parallel | dummy_parallel | dummy_lm]
        buffer_size: the reader will read this many lines from the text files and bufferes them before returning each
        max_length: for word-level it's better to be around 50-60, for bpe level around 128
        source_lang: the bi-letter tag indicating the source language ['en'|'fr'|'de'|...]
        target_lang: the bi-letter tag indicating the target language ['en'|'fr'|'de'|...]
        working_dir: the releatieve/absolute path of the dataset files
        train_file_name: the name of train files without the language extension
        test_file_name: the name of test files without the language extension
        dev_file_name: the name of dev files without the language extension
        granularity: 
            src: indicating the requested granularity level of the resulting dataset for source side as either of CHAR/BPE/WORD; possible values ["WORD" (default) | "BPE" | "CHAR"]
            tgt: indicating the requested granularity level of the resulting dataset for target side as either of CHAR/BPE/WORD; possible values ["WORD" (default) | "BPE" | "CHAR"]
        dummy: Only needed if you want to use the dummy data providers
            min_len: minimum length of the generated dummy sentences
            max_len: maximum length of the generated dummy sentences
            vocab_size: size of the generated vocabulary tokens
            train_samples: number of generated sequences used as train data
            test_samples: number of generated sequences used as test data
            dev_samples: number of generated sequences used as dev data
    vocab:
        bos_word: the special begin of sentence token
        eos_word: the special end of sentence token
        pad_word: the special pad token
        space_word: the special space token
        unk_word: the special unknown token
        bpe_separator: the special word-piece identifier token
        bpe_merge_size:
            src: # the expected number of merge operations if performing bpe on source side
            tgt: # the expected number of merge operations if performing bpe on target side
        min_count:
            src: # the minimum number of expected occurances in the source train data so that the token is not replaced with UNK token
            tgt: # the minimum number of expected occurances in the target train data so that the token is not replaced with UNK token
    preprocess:
        dataset_directory: the path to the directory containing raw data files
        result_directory: the path to the directory to which the results are to be stored (if not existing will be created)
        source_lang: the bi-letter tag indicating the source language ['en'|'fr'|'de'|...]
        target_lang: the bi-letter tag indicating the target language ['en'|'fr'|'de'|...]
        dataset_type: the type of the data placed inside dataset_directory; possible values [REGULAR | IWSLT]
trainer:
    model:
        ####### universal configurations
        type: possible values [seq2seq | rnnlm | transformer]
        bsize: size of the training sentence batches
        init_val: the value to range of which random variables get initiated in NN models
        best_model_path: the path to the best previously trained model (ending in ".pt"). This attribute is optional and if is not passes the defalt will be set to "None" for it
        decoder_weight_tying: whether the weights need to be tied between the decoder embedding and generator
        ####### seq2seq/rnnlm configurations
        tfr: teacher forcing ratio (if 1< teacher forcing is not used)
        bienc: bidirectional encoding (true or false)
        hsize: hidden state size of RNN layers
        nelayers: number of hidden layers in encoder
        ndlayers:  number of hidden layers in decoder
        ddropout: the dropout probability in the decoder
        ####### transformer configurations
        N: number of encoder/decoder layers
        d_model: size of each encoder/decoder layer
        d_ff: size of intermediate layer in feedforward sub-layers
        h: number of heads
        dropout: the dropout used in encoder/decoder model parts
        smoothing: the smoothing probability used in the genrating the output distribution (Label Smoothing technique)
        ####### bytenet configurations
        d: number of features in network
        max_r: max dilation size
        n_sets: number of ResBlock sets
        k: kernel size
    optimizer:
        name: possible values [adam | adadelta | sgd] # you can add other methods by modifying `create_optimizer` function in `translate.learning.estimator`
        lr: th initial learning rate
        gcn: grad clip norm value
        epochs: number of training epochs
        save_best_models: the feature of saving best found models while training (best based on train/dev loss) can be turned on/off using this feature
        early_stopping_loss: if the model reaches a loss below this value, the training will not continue anymore
        scheduler:
            name: possible values [cosine | step]
            eta_min: the lowest learninig rate value which the scheduler is allowed to decay the learning rate into (used for "cosine" scheduler)
            step_size: the step bucket size after which learning rate is multiplied by gamma(used for "step" scheduler)
            gamma: the learning rate multiplier after step_size number of steps (used for "step" scheduler)
        ####### transformer configurations
        warmup_steps: number of warmup steps before reaching the maxmimum learning rate 
        lr_update_factor: the lr factor suggested in "attention is all you need" paper
        needs_warmup: using the warmup wrapper feature can be turned on or off using this feature
        d_model: the size of model used in the warmup updates of the learning rate
    experiment:
        name: the experiment name which will be used when saving the best models
```

# Experiment Results
In this section, we put the experiment results of different models on different datasets. Please check this page regularly as we add our new results below the previously posted ones.
 If you are in Natlang Lab or have access to internal SFU servers, you can pick the pre-trained models the result of which is posted here from `/cs/natlang-expts/hassan/SFUTranslate/pretrained/`.
 Each experiment will have a model file (ending in ".pt") in there with the exact same name mentioned in the table below.
 The dataset with which the model has been trained is put in a folder besides the model with the exact same name as the model.
  The configuration file with which the model was configured, can be downloaded by clicking on the experiment name link (first column of the table).

|                                      Experiment Name                                      	|    Model    	|     Task     	|   Dataset   	|    Language    	| Greedy Decoding Test Score 	|                      More Info                      	|
|:-----------------------------------------------------------------------------------------:	|:-----------:	|:------------:	|:-----------:	|:--------------:	|:--------------------------:	|:---------------------------------------------------:	|
|     [seq2seq_reverse_copy](resources/documents/exp-configs/seq2seq_reverse_copy.yaml)     	|   Seq2Seq   	| SequenceCopy 	| ReverseCopy 	|   ReverseCopy  	|           99.703           	|             \|V\| = 1K; \|MaxLen\| = 50             	|
| [transformer_reverse_copy](resources/documents/exp-configs/transformer_reverse_copy.yaml) 	| Transformer 	| SequenceCopy 	| ReverseCopy 	|   ReverseCopy  	|           99.862           	|             \|V\| = 1K; \|MaxLen\| = 50             	|
|         [seq2seq_iwslt_de_en](resources/documents/exp-configs/seq2seq_de_en.yaml)         	|   Seq2Seq   	|  Translation 	|  IWSLT2017  	| German2English 	|           19.702           	| lowercased - tokenized with SpaCy; \|MaxLen\| = 100 	|
|     [transformer_iwslt_de_en](resources/documents/exp-configs/transformer_de_en.yaml)     	| Transformer 	|  Translation 	|  IWSLT2017  	| German2English 	|           23.052           	| lowercased - tokenized with SpaCy; \|MaxLen\| = 100 	|


# Requirements and Dependencies
In this section we go over the required libraries and how the project is dependant on each so that in case of the need 
to change (or remove) any of them, you know how and where to look for them.
The descriptions here are essentially describing the content of `requirements.txt` file besides this Readme document.

- [`PyYaml`](https://pyyaml.org/) the library needed to read the configurations, parse them and access their parsed values.
You may need to look at the content of `translate.configs.loader` class for the use cases of this library.

- [`PyTorch`](https://pytorch.org/docs/stable/index.html) the backend library which provides the neural network related 
functionality and classes. The only script accessing this library is `translate.backend.utils` which renames the `torch`
object as `backend` and every other script in the project accesses the `backend` object.
 If you prefer other NN frameworks (e.g. [`Tensorflow`](https://www.tensorflow.org/) or 
 [`DyNet`](https://dynet.readthedocs.io/en/latest/)) you can simply search for the occurances of the `backend` object 
  and update them to the way your desired framework does that task.

- [`spaCy`](https://spacy.io/) the pre-processing toolkit used for normalization and tokenization of `English`, `German`
, `Spanish`, `Portuguese`, `French`, `Italian`, and `Dutch`. However, to make the library able to process each of the 
languages you will need to download its resources for spaCy using the following lines (you should simply copy the 
download line and past it into the command line to get executed). You may need to look at the content of 
`translate.readers.tokenizer` class for the use cases of this library.
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
[Neural Machine Translation of Rare Words with Subword Units](http://www.aclweb.org/anthology/P16-1162) used in
`translate.readers.datareader` class for providing the Byte-Pair level granularity.

- Utility libraries \[`tqdm` and `abc` and `xml`\] the libraries that provide simple utility functionalities.

 
# FAQ

1. **What should I do if the `dev` score is going up but my `test` score is low?**  
check the test set files and make sure they are parallel in every line. If that is not the problem, you might have a vocabulary problem. If the distribution of the words in train and dev sets are closer but far away from the test set, the number of unknown words in the test set might go up, which immediately results in really low scores.  
Also try filtering out the non-frequent words in the train set to help model learn to deal with the unknown words (you can set the `reader.vocab.min_count` parameters to values grater than 1 to filter the less frequent words based on their train set frequency).

# Help and Comments
If you need help regarding the toolkit or you want to discuss your comments, you are more than welcome to email [Hassan S.Shavarani](sshavara@sfu.ca).

# Acknowledgments

We acknowledge and thank following people for using their work in this toolkit.

- We acknowledge Sean Robertson as the base implementation of the modules in `translate.learning.modules.rnn` is taken from his tutorial: 
  > [Translation with a Sequence to Sequence Network and Attention](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)    

- We acknowledge Alexander Rush as the base implementation of the modules in `translate.learning.modules.transformer` is taken from his tutorial:
  > [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
  
- We acknowledge David Pollack as the base implementation of the modules in `translate.learning.modules.cnn` is taken from his repository:
  > [bytenet.pytorch](https://github.com/dhpollack/bytenet.pytorch)
 