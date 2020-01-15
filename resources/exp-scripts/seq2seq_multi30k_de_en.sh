#! /bin/bash
pip install virtualenv
virtualenv sfutranslate
source sfutranslate/bin/activate
git clone https://github.com/sfu-natlang/SFUTranslate.git
cd SFUTranslate/ || return
git checkout 9a49ef584720db21ab457fd5da646da48689ebf6
python setup.py install
python -m spacy download en
python -m spacy download de
cd translate/ || return
python trainer.py ../resources/nmt.yml 2>train_progress_bars.log >train_output.log
python test_trained_model.py ../resources/nmt.yml 2>test_progress_bars.log >test_output.log
cat test_output.log # The dev and test scores are printed after this line
deactivate