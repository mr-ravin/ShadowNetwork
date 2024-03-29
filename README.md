## ShadowNetwork
This repository contains the source code and experimental results of the paper: "A new approach to train models based on LSTMs, GRUs, and other similar variants". The research paper is published in the CCIS Series of Springer.

#### Paper Title: A New Approach to Train LSTMs, GRUs, and Other Similar Networks for Data Generation

#### Author: [Ravin Kumar](https://mr-ravin.github.io)

#### Publication: 24th November 2019

#### Publication Link: https://link.springer.com/chapter/10.1007%2F978-981-15-1718-1_14

#### Doi: https://doi.org/10.1007/978-981-15-1718-1_14

#### Cite as:
```
Kumar R. (2019) A New Approach to Train LSTMs, GRUs, and Other Similar Networks for Data Generation. 
In: Prateek M., Sharma D., Tiwari R., Sharma R., Kumar K., Kumar N. (eds) Next Generation Computing 
Technologies on Computational Intelligence. NGCT 2018. Communications in Computer and Information Science, 
vol 922. Springer, Singapore
```

### Application Of this reseach work
A twitter bot is build by Author of this research work to show effectiveness of his proposed work in data generation, and named the bot [Heilige Quotes](https://twitter.com/HeiligeQuotes). This bot is deployed on Odroid XU4 computing device, and generate quotes for given hashtags, and then post them on its twitter handle. Source code for this bot is not released, as it could be used by some people for spamming on twitter. Although the deep learning architecture on which it is based on, is implemented in Tensorflow and provided in this repository. 

[![Working Demonstration](https://github.com/mr-ravin/ShadowNetwork/blob/master/heiligequotes.gif)](https://www.youtube.com/watch?v=FfgvSfScHR8)

### Project Structure
This repository contains the source code of "Shadow Network" over LSTMS and GRUs. implemented in Tensorflow over Simpson Dataset. The directory structure is as follows:
```
ShadowNetwork
|-LSTM
|   |-Uni (contains LSTM model without Shadow)
|   |-ShadowBoard (contains LSTM model with Shadow)
|   |-plot.pyc (python file to plot comparative graph)
|
|-GRU
|  |-Uni (contains GRU model without Shadow)
|  |-ShadowBoard (contains GRU model with Shadow)
|  |-plot.pyc (python file to plot comparative graph)
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
This will represent the 
ative difference between the training loss among model that uses 'Shadow' vs. that does not uses 'Shadow'.

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

```python

Copyright (c) 2017 Ravin Kumar
Website: https://mr-ravin.github.io

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation 
files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, 
modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the 
Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, 
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```
