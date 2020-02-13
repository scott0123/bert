# download the raw files
curl http://phontron.com/data/topicclass-v1.tar.gz -o topicclass.tar.gz
tar -xf topicclass.tar.gz
rm topicclass.tar.gz
mv topicclass dataset
# the validation data label needs to be fixed
sed -i 's/darama/drama/g' data/topicclass_valid.txt
# download the pretrained models
mkdir pretrained
curl https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip -o temp.zip
sudo apt-get -y install unzip
unzip temp.zip
mv cased_L-12_H-768_A-12 pretrained/cased_L-12_H-768_A-12
rm temp.zip
# create the model dir
mkdir output
