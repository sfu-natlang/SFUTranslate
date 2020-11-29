# You need to install boost_1_64_0 from the INSTALL directions in https://www.boost.org/users/history/version_1_64_0.html
# You need to install SRILM from http://www.speech.sri.com/projects/srilm/download.html
# git checkout https://github.com/moses-smt/mosesdecoder
# Install moses using the following command:
#	./bjam --with-srilm=/hassan_data/Software/srilm-1.7.3/ --with-boost=/hassan_data/Software/boost_1_64_0 -j8
export MOSESDECODER=<mosesdecoder_path>
export MOSES=$MOSESDECODER/bin/moses
export PATH=$PATH:<mosesdecoder_path>/bin/:<srilm_path>/srilm-1.7.3/bin/i686-m64:<srilm_path>/srilm-1.7.3/bin/
$MOSESDECODER/scripts/tokenizer/escape-special-chars.perl < $1/$1 > $1/$1.escape
$MOSESDECODER/scripts/recaser/train-recaser.perl --corpus $1/$1.escape --dir <absolute_path_to_the_passed_file>/$1  --lm=SRILM
rm $1/$1.escape
