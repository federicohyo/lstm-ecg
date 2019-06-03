# ecg QRS annotation (and segmentation) of the electrocardiogram (ECG) with a long short-term memory neural network. 

 
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
model.add(Dense(dimout, activation='softmax'))
adam = optimizers.adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy']) 
```

## Getting Started
- Download ECG /EDB data using something like <code>wget -r -l1 --no-parent https://physionet.org/physiobank/database/edb/</code>
- Run, with as the first argument the directory where the ECG data is stored; or set <code>qtdbpath</code>.



## Dependencies
I haven't got a list of all dependencies; but use this:  
- wfdb 1.3.4 ( not the newest >2.0, thanks BrettMontague); pip install wfdb==1.3.4
- tensorflow 
- keras
- numpy

