## ShadowNetwork
This repository contains the source code and experimental results of the paper.

Author: Ravin Kumar

### Application Of this reseach work
- A twitter bot is build by Author of this research work, and named the bot [Heilige Quotes](https://twitter.com/HeiligeQuotes).


### Project Structure
This repository contains the source code of "Shadow Network" over LSTMS and GRUs. implemented in Tensorflow over Simpson Dataset. The directory structure is as follows:
```
ShadowNetwork
|-LSTM
|   |-Uni (contains LSTM model without Shadow)
|   |-ShadowBoard (contains LSTM model with Shadow)
|   |-plot.pyc (python file to plot comparitive graph)
|
|-GRU
|  |-Uni (contains GRU model without Shadow)
|  |-ShadowBoard (contains GRU model with Shadow)
|  |-plot.pyc (python file to plot comparitive graph)
|
|-Results (Contains Experimental Result Data)
   |
   |-LSTM
   |  |
   |  |-Dropout_00_percent
   |  |-Dropout_25_percent
   |  |-Dropout_50_percent
   |
   |-GRU
      |
      |-Dropout_00_percent
      |-Dropout_25_percent
      |-Dropout_50_percent
```
#### Required Setups and Libraries
- Python3
- Tensorflow-1.2
- Numpy
- Matplotlib

## Steps for training LSTM with ShadowNetwork 
- Go to LSTM directory
- To train using Shadow , Go to ShadowBoard and,
- open terminal and type:
```
>>> python3 train.pyc
```
- it will train the LSTM based model using Shadow Network.
- Shadow.txt file contains the details regarding the training loss.

### Steps for training LSTM without ShadowNetwork
- Go to LSTM directory
- To train without Shadow , Go to Uni and,
- open terminal and type:
```
>>> python3 train.pyc
```
- it will train the LSTM based model without using Shadow Network.
- Uni.txt file contains the details regarding the training loss.

Now, Copy both "Uni.txt" and "Shadow.txt" to GRU directory, and run:
```
>>> python3 plot.pyc
```
This will represent the comparative difference between the training loss among model that uses 'Shadow' vs. that does not uses 'Shadow'.


### Steps for training GRU with ShadowNetwork 
- Go to GRU directory
- To train using Shadow , Go to ShadowBoard and,
- open terminal and type:
```
>>> python3 train.pyc
```
- it will train the GRU based model using Shadow Network.
- Shadow.txt file contains the details regarding the training loss.

### Steps for training GRU without ShadowNetwork
- Go to GRU directory
- To train without Shadow , Go to Uni and,
- open terminal and type:
```
>>> python3 train.pyc
```
- it will train the GRU based model without using Shadow Network.
- Uni.txt file contains the details regarding the training loss.

Now, Copy both "Uni.txt" and "Shadow.txt" to LSTM directory, and run:
```
>>> python3 plot.pyc
```
This will represent the comparative difference between the training loss among model that uses 'Shadow' vs. that does not uses 'Shadow'.

### Inference for Uni model
```
>>> python3 infer.pyc
Select Topic:  # this will act as starting word (input)
joy
```
### Inference for ShadowBoard model
```
>>> python3
>>> import infer
>>> t= infer.run("","joy",200) # here joy is the starting word needed for starting inference.
>>> print(t)
```
### Experimental Result Data:
Inside "Results" directory, both LSTM and GRU directories contains folders, Having results during different dropouts :
- Dropout_00_percent
- Dropout_25_percent
- Dropout_50_percent

The "Shadow.txt" contains the training loss when Shadow Network is applied, while "Uni.txt" contains the training loss of the model "without Shadow Network".To Generate the visual representation, go inside the Dropout_XX_percent directories, and run the following command:
```
>>> python3 plot.py
```

### Note: This work can be used only for academic research work after providing proper citation and deserved credits to this work. For Industrial, commercial or any other use, permission is required from the Author.
