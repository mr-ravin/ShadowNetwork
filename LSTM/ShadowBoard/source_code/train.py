import tensorflow as tf
import helper
import numpy as np
from collections import Counter
from tensorflow.contrib import seq2seq
from time import gmtime, strftime
import matplotlib.pyplot as plt
f_cost="shadow.txt"
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
lstm_layers=2
cost_list=[]
cost_list_f=[]
#START######Load Data ############################
data_dir = './data/'
print("enter the name of training data file(.txt) :")
tdf=str(input())
#tdf="simpson.txt"
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


#### building the Graph for training ####

train_graph = tf.Graph()
with train_graph.as_default():
  
  
  vocab_size=len(int_to_vocab)
  input_text=tf.placeholder(tf.int32,[None,None],name="input")
  input_text_rev=tf.placeholder(tf.int32,[None,None],name="input_rev")
  targets=tf.placeholder(tf.int32,[None,None],name="targets")
  targets_rev=tf.placeholder(tf.int32,[None,None],name="targets_rev")
  #lr=tf.placeholder(tf.float32,name="learning_rate")
  #lr_f=tf.placeholder(tf.float32,name="learning_rate_f")
  input_data_shape = tf.shape(input_text)
  input_data_shape_rev=tf.shape(input_text_rev)

  #lstm_layers=3

  lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
  drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=0.5)
  cell = tf.contrib.rnn.MultiRNNCell([drop] * lstm_layers,)
  initial_st = cell.zero_state(input_data_shape[0], tf.float32,)
  initial_state=tf.identity(initial_st,name="initial_state")


  lstm_rev = tf.contrib.rnn.BasicLSTMCell(rnn_size)
  drop_rev = tf.contrib.rnn.DropoutWrapper(lstm_rev, output_keep_prob=0.5)
  cell_rev = tf.contrib.rnn.MultiRNNCell([drop_rev] * lstm_layers,)
  initial_st_rev=cell_rev.zero_state(input_data_shape_rev[0], tf.float32,)
  initial_state_rev=tf.identity(initial_st_rev,name="initial_state_rev")


  embedding = tf.Variable(tf.random_uniform((vocab_size,rnn_size), -1, 1)) # get embedding

  embed = tf.nn.embedding_lookup(embedding, input_text)
  embed_rev=tf.nn.embedding_lookup(embedding, input_text_rev) ##here
  outputs, final_st = tf.nn.dynamic_rnn(cell, embed,dtype=tf.float32) # RNN
  with tf.variable_scope('System_LSTM'):
    outputs_rev,final_st_rev=tf.nn.dynamic_rnn(cell_rev, embed_rev,dtype=tf.float32) # RNN
  
  final_st=tf.identity(final_st,name="final_state")
  final_st_rev=tf.identity(final_st_rev,name="final_state_rev")
  #outputs_tmp=outputs_rev  
  outputs_tmp=tf.concat([outputs,outputs_rev],2)

  logits = tf.contrib.layers.fully_connected(outputs_tmp,vocab_size, 
                                               weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                               biases_initializer=tf.zeros_initializer(),
                                               activation_fn=None)

  logits_f=tf.contrib.layers.fully_connected(outputs,vocab_size, 
                                               weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                               biases_initializer=tf.zeros_initializer(),
                                               activation_fn=None)

  probs = tf.nn.softmax(logits, name='probs')
  probs_f=tf.nn.softmax(logits_f, name='probs_f')

  cost = seq2seq.sequence_loss(logits,targets,tf.ones([input_data_shape[0], input_data_shape[1]])) #tf.ones used to build weight matrix
  cost_f = seq2seq.sequence_loss(logits_f,targets,tf.ones([input_data_shape[0], input_data_shape[1]])) #tf.ones used to build weight matrix  
 
  optimizer = tf.train.AdamOptimizer()
  optimizer_f = tf.train.AdamOptimizer()
  """
  gradients = optimizer.compute_gradients(cost)
  gradients_f=optimizer_f.compute_gradients(cost_f)

  capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients]
  train_op = optimizer.apply_gradients(capped_gradients)
  
  capped_gradients_f = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients_f]
  train_op_f = optimizer_f.apply_gradients(capped_gradients_f)
  """
  train_op=optimizer.minimize(cost)
  train_op_f=optimizer_f.minimize(cost_f)
#### building the Graph for training ####

start_time=strftime("%Y-%m-%d %H:%M:%S", gmtime())
#START################ TRAINING MODEL #######################



num_batch=(len(int_text)-1)//(batch_size*seq_length)
x= int_text[:num_batch*batch_size*seq_length]
y=int_text[1:num_batch*batch_size*seq_length+1]
batches = np.zeros((num_batch,2,batch_size,seq_length),dtype=int)
#batches_rev=np.zeros((num_batch,2,batch_size,seq_length),dtype=int)
index = 0
for n in range(num_batch):
    for b in range(batch_size):
        batches[n][0][b] = np.array(x[index:index+seq_length])
        #tmp_x=x[index:index+seq_length]
        #tmp_x.reverse()
        #batches_rev[n][0][b] = np.array(tmp_x)

        batches[n][1][b] = np.array(y[index:index+seq_length])

        #tmp_y=y[index:index+seq_length]
        #tmp_y.reverse()
        #batches_rev[n][1][b] = np.array(tmp_y)

        index = index+seq_length


with tf.Session(graph=train_graph) as sess:
    writer=tf.summary.FileWriter("./graph",sess.graph)
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(num_epochs):
        state = sess.run(initial_state, {input_text: batches[0][0]})
        state_rev=sess.run(initial_state, {input_text: batches[0][-1]})
        
        for batch_i, (x, y) in enumerate(batches):
            x_rev=np.flip(x,1)
            y_rev=np.flip(y,1)
            feed_tmp = {
                input_text: x,
                input_text_rev: x_rev,
                targets: y,
                targets_rev: y_rev,
                initial_state: state,
                initial_state_rev: state_rev}#,
                #lr: learning_rate}
            feed_tmp_f = {
                input_text: x,
                targets: y,
                initial_state: state}#,
                #lr_f: learning_rate_f}

            train_loss, state, _ = sess.run([cost, final_st, train_op], feed_tmp) ###
            cost_list.append(train_loss)
            train_loss_f, state_f, _ = sess.run([cost_f, final_st, train_op_f], feed_tmp_f)#feed_dict={ input_text: x,targets: y,initial_state: state,lr_f: learning_rate}) ###
            
            cost_list_f.append(train_loss_f)
        
            if (epoch_i * len(batches) + batch_i) % show_every_n_batches == 0:
                print('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                    epoch_i,
                    batch_i,
                    len(batches),
                    train_loss))
                print('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                    epoch_i,
                    batch_i,
                    len(batches),
                    train_loss_f))

    # Save Model
    saver = tf.train.Saver()
    saver.save(sess, save_dir)
    print('Model Trained and Saved')
writer.close()
#END################ TRAINING MODEL ########################
fin_time=strftime("%Y-%m-%d %H:%M:%S", gmtime())
print(start_time)
print("\n",fin_time)

y_len=[]
for i in range(len(cost_list)):
  y_len.append(i)

#START################### Model Parameter Saved ############

helper.save_params((seq_length, save_dir))

#END##################### Model Parameter Saved ############
tmp=""
for i in cost_list_f:
  tmp=tmp+" "+str(i)
tmp=tmp[1:] 
f_wr.write(tmp)
f_wr.close()
 
plt.plot(cost_list_f,y_len)
plt.xlabel("Step")
plt.ylabel("Cost of Shadow")
plt.savefig("shadowCost.png")
plt.show()
