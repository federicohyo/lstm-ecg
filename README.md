# Real Time Electrocardiogram Annotation with a Long Short Term Memory Neural Network 

Here you will find code that describes a neural network model capable of labeling the R-peak of ECG recordings. This method has been tested on a wearable device as well as with public datasets.

This code trains a neural network with a loss function that maximizes F1 score (binary position of peak in a string of 0's and 1's.).

The model demonstrates high accuracy of in labeling the R-peak of QRS complexes of ECG signal of public available datasets (MITDB and EDB). 
Results are compared with the gold standard method Pan-Tompkins. Our method demonstrates superior generalization performance across different datasets. 

The network has been validated with data using an IMEC wearable device on an elderly population of patients which all have heart failure and co-morbidities. This demonstrates that the proposed solution is capable of performing close to human annotation 94.8% average accuracy, on single lead wearable data containing a wide variety of QRS and ST-T morphologies. 


## Model

```
model = Sequential()
model.add(Dense(32,W_regularizer=regularizers.l2(l=0.01), input_shape=(seqlength, features)))
model.add(Bidirectional(LSTM(64, return_sequences=True)))#, input_shape=(seqlength, features)) ) ### bidirectional ---><---
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(64, activation='relu',W_regularizer=regularizers.l2(l=0.01)))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(dimout, activation='sigmoid'))
adam = optimizers.adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='f1_loss', optimizer=adam, metrics=['f1_m, precision_m, recall_m, matthews_correlation']) 
```

## Getting Started
- Download ECG /EDB data using something like <code>wget -r -l1 --no-parent https://physionet.org/physiobank/database/edb/</code>
- Run, with as the first argument the directory where the ECG data is stored; or set <code>qtdbpath</code>.


## Dependencies
You will need the following packages:  
- wfdb 1.3.4 ( not the newest >2.0); pip install wfdb==1.3.4
- tensorflow 
- keras
- numpy
- scipy 

## Datasets

MIT-BIH Arrhythmia Database - https://physionet.org/content/mitdb/1.0.0/
Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG, Mietus JE, Moody GB, Peng C-K, Stanley HE. PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals (2003). Circulation. 101(23):e215-e220.

European ST-T Database - EDB
Taddei A, Distante G, Emdin M, Pisani P, Moody GB, Zeelenberg C, Marchesi C. The European ST-T Database: standard for evaluating systems for the analysis of ST-T changes in ambulatory electrocardiography. European Heart Journal 13: 1164-1172 (1992).

## Ipython notebook scripts:
   	train_lstm_edb.ipynb - works with European ST-T Dataset
    train_lstm_mitd.ipynb - works with MIT-BIH Arrhythmia Database 

## More info? Feel free to cite!
When using this resource, please cite the original publication: 
F. Corradi, J. Buil, H. De Canniere, W. Groenendaal, P. Vandervoort. "Real Time Electrocardiogram Annotation with a Long Short Term Memory Neural Network", 2019 IEEE Biomedical Circuits and Systems Conference (BioCAS) Proceedings, Nara, Japan. (in press)
