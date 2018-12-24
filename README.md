# SFUTranslate

This is an academic machine translation toolkit, in which the main focus has been towards readability and changeability.
We also have tried to make the algorithms as fast as possible, but please let us know if you have any suggestions or
concerns regarding the toolkit. To get familiar with what you can do and how you can do it please read through this documentation.

To run the code you will need python 3.5+ and PyTorch 0.4+.

Note: the base implementation of the modules in `translate.learning.modules.rnn` is taken from Sean Robertson's tutorial: 
> [Translation with a Sequence to Sequence Network and Attention](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)    
 
#Getting Started

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
  - `estimator` the script containing the `Estimator` class which will do forward and backward passes in to train the model and evaluate it.
  - `modelling` the abstract class which unifies the interface of models defined in the project. If you are making a new model (e.g. a sequence to sequence model or a language model) using the modules defined in the project (or the modules you have defined and added to the project), your model class needs to extend the `AbsCompleteModel` in this script.
- `logging` creates and provides a single instance of logger for all that needs to be logged across the project.

- `readers` provides all the funcionalities necessary for reading, understanding and processing the datasets. 
You may want to write your own dataset reader, in which case your dataset reader class must extend the abstact 
`AbsDatasetReader` class defined in this package. The dummy dataset providers `ReverseCopyDataset` and 
`SimpleGrammerLMDataset` are examples you can look at to understand how you may create your dataset provider.  

## Help and Comments
If you need help regarding the toolkit or you want to discuss your comments, you are more than welcome to email [Hassan S.Shavarani](sshavara@sfu.ca).