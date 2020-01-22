#! /bin/bash
pip3 install virtualenv
virtualenv sfutranslate
source sfutranslate/bin/activate
git clone https://github.com/sfu-natlang/SFUTranslate.git
cd SFUTranslate/ || return
git checkout 7b8fa3e5551f92007c0e9dc600b46bbb8d3f0fb8
python setup.py install
python -m spacy download en
python -m spacy download de
cd translate/ || return
python trainer.py ../resources/exp-configs/seq2seq_iwslt_de_en.yml 2>train_progress_bars.log >train.output
python test_trained_model.py ../resources/exp-configs/seq2seq_iwslt_de_en.yml 2>testerr.log >test.output
cat test.output # The dev and test scores are printed after this line
deactivate