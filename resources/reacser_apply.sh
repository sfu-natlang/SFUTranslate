# You need to install boost_1_64_0 from the INSTALL directions in https://www.boost.org/users/history/version_1_64_0.html
# You need to install SRILM from http://www.speech.sri.com/projects/srilm/download.html
# git checkout https://github.com/moses-smt/mosesdecoder
# Install moses using the following command:
#	./bjam --with-srilm=/hassan_data/Software/srilm-1.7.3/ --with-boost=/hassan_data/Software/boost_1_64_0 -j8
export MOSESDECODER=<mosesdecoder_path>
export MOSES=$MOSESDECODER/bin/moses
export PATH=$PATH:<mosesdecoder_path>/bin/:<srilm_path>/srilm-1.7.3/bin/i686-m64:<srilm_path>/srilm-1.7.3/bin/
$MOSESDECODER/scripts/tokenizer/escape-special-chars.perl < $1 > $1.escaped
$MOSESDECODER/scripts/recaser/recase.perl --in $1.escaped --model $2/moses.ini > $1.recased.escaped
$MOSESDECODER/scripts/tokenizer/deescape-special-chars.perl < $1.recased.escaped > $1.recased
rm $1.escaped
rm $1.recased.escaped

