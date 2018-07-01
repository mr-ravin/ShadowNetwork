import tensorflow as tf
import numpy as np
import helper
from collections import Counter
from tensorflow.contrib import seq2seq

# Number of Epochs
num_epochs = 100
# Batch Size
batch_size =500
# RNN Size
rnn_size = 512
# Sequence Length
seq_length = 15
# Learning Rate
learning_rate = 0.01
# Show stats for every n number of batches
show_every_n_batches = 50
# directory for saving the model
save_dir = './save'
# length of generated text
gen_length = 200
# Prime Word
print("Select Topic: ")
prime_word = str(input()) # Key word to be used to generate the text
#prime_word='moe_szyslak'
_, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()

seq_length, load_dir = helper.load_params()


def get_tensors(loaded_graph):
    
    InputTensor = loaded_graph.get_tensor_by_name('input:0')
    InitialStateTensor=loaded_graph.get_tensor_by_name('initial_state:0')
    FinalStateTensor=loaded_graph.get_tensor_by_name('final_state:0')
    ProbsTensor=loaded_graph.get_tensor_by_name('probs:0')
    
    return (InputTensor,InitialStateTensor,FinalStateTensor,ProbsTensor)

def pick_word(probabilities, int_to_vocab):
    
    index=np.argmax(probabilities)
    word=int_to_vocab[index]
    
    return word



loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # Load saved model
    loader = tf.train.import_meta_graph(load_dir + '.meta')
    loader.restore(sess, load_dir)

    # Get Tensors from loaded model
    input_text, initial_state, final_state, probs = get_tensors(loaded_graph)

    # Sentences generation setup
    gen_sentences = [prime_word] #changed
    prev_state = sess.run(initial_state, {input_text: np.array([[1]])})

    # Generate sentences
    for n in range(gen_length):
        # Dynamic Input
        dyn_input = [[vocab_to_int[word] for word in gen_sentences[-seq_length:]]]
        dyn_seq_length = len(dyn_input[0])
        print("dynamic length= ",dyn_seq_length)
        # Get Prediction
        probabilities, prev_state = sess.run(
            [probs, final_state],
            {input_text: dyn_input, initial_state: prev_state})
        print("probabilities= ",probabilities)
        print("length of probabilities =",len(probabilities))
        pred_word = pick_word(probabilities[0][dyn_seq_length-1], int_to_vocab)  # changed line !!! (it is the correct one)

        gen_sentences.append(pred_word)
    
    # Remove tokens
    quote_script = ' '.join(gen_sentences)
    for key, token in token_dict.items():
        ending = ' ' if key in ['\n', '(', '"'] else ''
        quote_script = quote_script.replace(' ' + token.lower(), key)
    quote_script = quote_script.replace('\n ', '\n')
    quote_script = quote_script.replace('( ', '(')
        
    print(quote_script)
