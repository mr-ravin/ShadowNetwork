import tensorflow as tf
import helper
import numpy as np
from collections import Counter
from tensorflow.contrib import seq2seq
from time import gmtime, strftime
import matplotlib.pyplot as plt
f_cost="uni.txt"
f_wr=open(f_cost,"a")
# Number of Epochs
num_epochs = 200
# Batch Size
batch_size =250
# RNN Size
rnn_size = 512
# Sequence Length
seq_length = 21
# Learning Rate
learning_rate = 0.001
# Show stats for every n number of batches
show_every_n_batches = 50
#directory for saving the model
save_dir = './save'
#length of generated text
gen_length = 200
#lstm layers
lstm_layers=3
cost_list=[]
#START######Load Data ############################
data_dir = './data/'
print("enter the name of training data file(.txt) :")
tdf=str(input())
data_dir=data_dir + tdf
text = helper.load_data(data_dir)
#END########Load Data ############################


def create_lookup_tables(text):
    
    count_list=Counter(text)
    vocab_to_int={}
    int_to_vocab={}
    
    for i in enumerate(count_list):
        vocab_to_int[str(i[1])]=int(i[0])

        int_to_vocab[int(i[0])]=str(i[1])

    return (vocab_to_int,int_to_vocab)



def token_lookup():
    
    dict={'.':'||period||',',':'||comma||','"':'||quote||',';':'||semicolon||','!':'||exclamation||','?':'||question||','(':'||leftpar||',')':'||rightpar||','--':'||dash||','\n':'||return||'}
    return dict


#START########## Preprocess and save the data ##########

helper.preprocess_and_save_data(data_dir, token_lookup, create_lookup_tables)

#END############ Preprocess and save the data ###########



#START######### LOAD THE SAVED DATA (PREPROCESSED) #######

int_text, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()

#END########### LOAD THE SAVED DATA (PREPROCESSED) #######

def get_inputs():
    
    inputs=tf.placeholder(tf.int32,[None,None],name="input")
    targets=tf.placeholder(tf.int32,[None,None],name="targets")
    learning_rate=tf.placeholder(tf.float32,name="learning_rate")    
    return (inputs,targets,learning_rate)



def get_init_cell(batch_size, rnn_size):
    
    #lstm_layers=3
    lstm = tf.contrib.rnn.GRUCell(rnn_size)
    drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=0.5)
    #cell = tf.contrib.rnn.MultiRNNCell([drop])#([drop] * lstm_layers,)
    cell = tf.contrib.rnn.MultiRNNCell([drop] * lstm_layers,)
    initial_st = cell.zero_state(batch_size, tf.float32,)
    initial_state=tf.identity(initial_st,name="initial_state")
    return (cell,initial_state)



def get_embed(input_data, vocab_size, embed_dim):
   
    embedding = tf.Variable(tf.random_uniform((vocab_size,embed_dim), -1, 1))
    embed = tf.nn.embedding_lookup(embedding, input_data)
    return embed



def build_rnn(cell, inputs):

    outputs, final_st = tf.nn.dynamic_rnn(cell, inputs,dtype=tf.float32)
    final_state=tf.identity(final_st,name="final_state")
    return (outputs,final_state)



def build_nn(cell, rnn_size, input_data, vocab_size):
    
    embed=get_embed(input_data,vocab_size,rnn_size)
    outputs, final_state = build_rnn(cell,embed)
    logits = tf.contrib.layers.fully_connected(outputs,vocab_size, 
                                               weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                               biases_initializer=tf.zeros_initializer(),
                                               activation_fn=None)
    return (logits,final_state)



def pick_word(probabilities, int_to_vocab):
    """
    return: String of the predicted word
    """
    index=np.argmax(probabilities)
    word=int_to_vocab[index]   
    return word



def get_batches(int_text, batch_size, seq_length):
    
    num_batch=(len(int_text)-1)//(batch_size*seq_length)
    x= int_text[:num_batch*batch_size*seq_length] 
    y=int_text[1:num_batch*batch_size*seq_length+1]
    batch = np.zeros((num_batch,2,batch_size,seq_length),dtype=int)
    index = 0
    for n in range(num_batch):
        for b in range(batch_size):
            batch[n][0][b] = np.array(x[index:index+seq_length])
            batch[n][1][b] = np.array(y[index:index+seq_length])
            index = index+seq_length
    
    return batch




#### building the Graph for training ####

train_graph = tf.Graph()
with train_graph.as_default():
  vocab_size = len(int_to_vocab)
  input_text, targets, lr = get_inputs()
  input_data_shape = tf.shape(input_text)
  cell, initial_state = get_init_cell(input_data_shape[0], rnn_size) #input_data_shape[0] represents number of batches
  logits, final_state = build_nn(cell, rnn_size, input_text, vocab_size)
  # Probabilities for generating words
  probs = tf.nn.softmax(logits, name='probs')

  # Loss function
  cost = seq2seq.sequence_loss(logits,targets,tf.ones([input_data_shape[0], input_data_shape[1]])) #tf.ones used to build weight matrix

  # Optimizer
  optimizer = tf.train.AdamOptimizer(lr)
  val=optimizer.minimize(cost)
  # Gradient Clipping
  ##gradients = optimizer.compute_gradients(cost)
  ##capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients]
  ##train_op = optimizer.apply_gradients(capped_gradients)

#### building the Graph for training ####


starttime=strftime("%Y-%m-%d %H:%M:%S", gmtime())
#START################ TRAINING MODEL #######################

batches = get_batches(int_text, batch_size, seq_length)
with tf.Session(graph=train_graph) as sess:
    writer=tf.summary.FileWriter("./graph",sess.graph)
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(num_epochs):
        state = sess.run(initial_state, {input_text: batches[0][0]})

        # loop to train on each batch
        for batch_i, (x, y) in enumerate(batches):
            feed = {
                input_text: x,
                targets: y,
                initial_state: state,
                lr: learning_rate}
            train_loss, state, _ = sess.run([cost, final_state, val], feed)
            cost_list.append(train_loss)
            # Show every <show_every_n_batches> batches
            if (epoch_i * len(batches) + batch_i) % show_every_n_batches == 0:
                print('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                    epoch_i,
                    batch_i,
                    len(batches),
                    train_loss))

    # Save Model
    saver = tf.train.Saver()
    saver.save(sess, save_dir)
    print('Model Trained and Saved')
writer.close()
#END################ TRAINING MODEL ########################
fintime=strftime("%Y-%m-%d %H:%M:%S", gmtime())
print(starttime)
print("\n",fintime)

y_len=[]
for i in range(len(cost_list)):
  y_len.append(i)

#START################### Model Parameter Saved ############

helper.save_params((seq_length, save_dir))

#END##################### Model Parameter Saved ############
tmp=""
for i in cost_list:
  tmp=tmp+" "+str(i)
tmp=tmp[1:] 
f_wr.write(tmp)
f_wr.close()


plt.plot(cost_list,y_len)
plt.xlabel("Step")
plt.ylabel("Cost of Uni")
plt.savefig("UniCost.png")
plt.show()
