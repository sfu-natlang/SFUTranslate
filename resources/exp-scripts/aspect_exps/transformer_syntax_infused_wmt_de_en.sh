#! /bin/bash
wget https://www.python.org/ftp/python/3.8.2/Python-3.8.2.tgz
tar zxvf Python-3.8.2.tgz
cd Python-3.8.2
./configure --prefix=`pwd`
make
make install
cd ..
Python-3.8.2/bin/python3.8 -m pip install virtualenv
Python-3.8.2/bin/virtualenv -p Python-3.8.2/bin/python3.8 sfutranslate
source sfutranslate/bin/activate
export PYTHONPATH=sfutranslate/lib/python3.8/site-packages
git clone -b master https://github.com/sfu-natlang/SFUTranslate.git
cd SFUTranslate/ || return
git checkout 0000000000000000000000000000000000000000
python setup.py install
# python -m spacy download en_core_web_lg
python -m spacy download de_core_news_md
cd translate/ || return
# WARNING change the number of visible GPU if GPU:0 is already allocated
export CUDA_VISIBLE_DEVICES=0
echo "Starting to train the model, you can check the training process by running the following command in SFUTranslate/translate directory (however, do not kill this process)"
echo "    tail -f train_progress_bars.log"
python trainer.py ../resources/exp-configs/aspect_exps/transformer_syntax_infused_wmt19_de_en.yml 2>train_progress_bars.log >train.output
echo "Starting to test the best trained model, you can find the test results in \"test.output\" in SFUTranslate/translate directory"
python test_trained_model.py ../resources/exp-configs/aspect_exps/transformer_syntax_infused_wmt19_de_en.yml 2>testerr.log >test.output
cat test.output # The dev and test scores are printed after this line
deactivate