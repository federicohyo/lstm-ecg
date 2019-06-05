# R-PEAK and QRS annotation of electrocardiogram (ECG) signal with a long short-term memory neural network. 

## DISCLAIMER

The complete code will be committed after the review process.. ( to be entered after acceptance). Please stay tuned!
 
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
