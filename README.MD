# Coding up a Neural Network classifier from scratch

<p align="center">
<img src="./fig/LSTM-CRF.png" width="50%">
</p>
 
We train a bidirectional LSTM network with an additional biLSTM layer for character embedding processing as well as feature augmentation. The 
network's hyperparameters are loaded from the config files: `./config/config*.json`

Results are dump in the `/results` folder, and the model summary together with the architecture, weights and indexes are output in `/models`. 

Training, evaluation and test data are within the `/data` folder (you have to copy it there), which also contains an `/xml2json` folder containing the output from processing 
all the `.xml` files.

If you want to run the code with the pre-trained embedding vectors option, you have to download the pre-trained vectors from the GloVe webpage. A link to the data sets can be found in 
the jupyter notebook within /report.

### Usage

Run:
```
/src/nnerc.py 
```

Make sure that within this file, the version number matches the desired config file.