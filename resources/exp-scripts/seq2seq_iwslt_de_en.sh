#! /bin/bash
wget https://www.python.org/ftp/python/3.5.2/Python-3.5.2.tgz
tar zxvf Python-3.5.2.tgz
cd Python-3.5.2
./configure --prefix=`pwd`
make
make install
cd ..
Python-3.5.2/bin/python3.5 -m pip install virtualenv
Python-3.5.2/bin/virtualenv -p Python-3.5.2/bin/python3.5 sfutranslate
source sfutranslate/bin/activate
export PYTHONPATH=sfutranslate/lib/python3.5/site-packages
git clone -b master https://github.com/sfu-natlang/SFUTranslate.git
cd SFUTranslate/ || return
git checkout 7957c261434bc0ea806ba750811d3a8030a510b9
python setup.py install
python -m spacy download en
python -m spacy download de
cd translate/ || return
echo "Starting to train the model, you can check the training process by running the following command in SFUTranslate/translate directory (however, fo not kill this process)"
echo "    tail -f train_progress_bars.log"
python trainer.py ../resources/exp-configs/seq2seq_iwslt_de_en.yml 2>train_progress_bars.log >train.output
echo "Starting to test the best trained model, you can find the test results in \"test.output\" in SFUTranslate/translate directory"
python test_trained_model.py ../resources/exp-configs/seq2seq_iwslt_de_en.yml 2>testerr.log >test.output
cat test.output # The dev and test scores are printed after this line
deactivate