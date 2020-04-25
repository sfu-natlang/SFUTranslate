#! /bin/bash
pip3 install virtualenv
virtualenv sfutranslate
source sfutranslate/bin/activate
export PYTHONPATH=sfutranslate/lib/python3.5/site-packages
git clone -b augmented https://github.com/sfu-natlang/SFUTranslate.git
cd SFUTranslate/ || return
git checkout b2482298ba57ae1a7c28a1e7423185a6fac82a11
python setup.py install
python -m spacy download en
python -m spacy download de
cd translate/ || return
echo "Starting to train the model, you can check the training process by running the following command in SFUTranslate/translate directory (however, fo not kill this process)"
echo "    tail -f train_progress_bars.log"
python trainer.py ../resources/exp-configs/transformer_multi30k_de_en.yml 2>train_progress_bars.log >train.output
echo "Starting to test the best trained model, you can find the test results in \"test.output\" in SFUTranslate/translate directory"
python test_trained_model.py ../resources/exp-configs/transformer_multi30k_de_en.yml 2>testerr.log >test.output
cat test.output # The dev and test scores are printed after this line
deactivate