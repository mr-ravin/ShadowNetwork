import tensorflow as tf
import numpy as np
import helper
import random
from collections import Counter
from tensorflow.contrib import seq2seq
def run(username,prime_word,gen_length):
 # directory for saving the model
 save_dir = './save'

 _, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()  

 seq_length, load_dir = helper.load_params()

 loaded_graph = tf.Graph()
 with tf.Session(graph=loaded_graph) as sess:
    # Load saved model
    loader = tf.train.import_meta_graph(load_dir + '.meta')
    loader.restore(sess, load_dir)

    # Get Tensors from loaded model
    input_text = loaded_graph.get_tensor_by_name('input:0')
    #input_text_rev=loaded_graph.get_tensor_by_name('input_rev:0')
    initial_state=loaded_graph.get_tensor_by_name('initial_state:0')
    final_state=loaded_graph.get_tensor_by_name('final_state:0')
    probs=loaded_graph.get_tensor_by_name('probs_f:0')

    # Sentences generation setup
    gen_sentences = [prime_word] #changed
    prev_state = sess.run(initial_state, {input_text: np.array([[1]])})#,input_text_rev: np.array([[1]])})

    # Generate sentences
    for n in range(gen_length):
        # Dynamic Input
        dyn_input = [[vocab_to_int[word] for word in gen_sentences[-seq_length:]]]
        dyn_seq_length = len(dyn_input[0])
        # Get Prediction
        probabilities, prev_state = sess.run(
            [probs, final_state],
            {input_text: dyn_input, initial_state: prev_state,})

        index=np.argmax(probabilities[0][dyn_seq_length-1]) ###
        pred_word=int_to_vocab[index]

        gen_sentences.append(pred_word)
    
    # Remove tokens
    quote_script = ' '.join(gen_sentences)
    for key, token in token_dict.items():
        ending = ' ' if key in ['\n', '(', '"'] else ''
        quote_script = quote_script.replace(' ' + token.lower(), key)
    quote_script = quote_script.replace('\n ', '\n')
    quote_script = quote_script.replace('( ', '(')

    quote_data=quote_script.split("\n")    
    quote_data=quote_data[1:-1]
    res=[]
    for i in quote_data:
      res.append(i)
    for i in quote_data:
      if len(i)<=139-len(username) and len(i)>9: # "i have" not in i:
        res.append(i) # remove it
    #final list of result is in "res"
    choice=random.randint(1,1000)
    choice_fin=choice%len(res)

    #return username+" "+res[choice_fin]
    return res
