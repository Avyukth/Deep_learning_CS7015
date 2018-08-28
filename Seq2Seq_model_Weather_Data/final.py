####### References Taken from 
######  https://github.com/udacity/deep-learning/blob/master/seq2seq/ #############################

import os
import numpy as np
import time
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple
import helper
from distutils.version import LooseVersion
import tensorflow as tf
from tensorflow.python.layers.core import Dense
from calbleu import *
import argparse
from pdb import set_trace as bp
from matplotlib import pyplot as plt


parser = argparse.ArgumentParser(description='sequence to sequence model')
parser.add_argument("--lr", default=0.001,type=float)
parser.add_argument("--batch_size", default=150 ,type=int)
parser.add_argument("--init", default=1)
parser.add_argument("--save_dir" ,default=os.getcwd(),type=str)
parser.add_argument("--decode_method", default="greedy",help="greedy or beam search method")
parser.add_argument("--beam_width", default=-5,type=int)

parser.add_argument('--use_dropout', type=float, default=0, 
                    help='input dropout default 0.5')
parser.add_argument('--use_residual', type=bool, default=False,
                    help='use residual network')
parser.add_argument('--earlystop', type=int, default=1,
                    help='set earlystop parameter')
parser.add_argument("--length_penalty_weight", type=float, default=0.0, help="Length penalty for beam search.")
args = parser.parse_args()


def extract_character_vocab(data):
    special_words = ['<PAD>', '<GO>',  '<EOS>']

    set_words = set([character for line in data.split('\n') for character in line.split()])
    int_to_vocab = {word_i: word for word_i, word in enumerate(special_words + list(set_words))}
    vocab_to_int = {word: word_i for word_i, word in int_to_vocab.items()}

    return int_to_vocab, vocab_to_int


"""### Input"""

def get_model_inputs():
    input_data = tf.placeholder(tf.int32, [None, None], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    lr = tf.placeholder(tf.float32, name='learning_rate')

    target_sequence_length = tf.placeholder(tf.int32, (None,), name='target_sequence_length')
    max_target_sequence_length = tf.reduce_max(target_sequence_length, name='max_target_len')
    source_sequence_length = tf.placeholder(tf.int32, (None,), name='source_sequence_length')
    
    return input_data, targets, lr, target_sequence_length, max_target_sequence_length, source_sequence_length

def encoding_layer(input_data, rnn_size, num_layers,
                   source_sequence_length, source_vocab_size, 
                   encoding_embedding_size):


    # Encoder embedding
    enc_embed_input = tf.contrib.layers.embed_sequence(input_data, source_vocab_size, encoding_embedding_size)

    # RNN cell
    def make_cell(rnn_size):
        enc_cell = tf.contrib.rnn.LSTMCell(rnn_size,
                                           initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        
        return enc_cell

    enc_cell=make_cell(rnn_size/2)
    

    ((encoder_fw_outputs,
      encoder_bw_outputs),
     (encoder_fw_final_state,
      encoder_bw_final_state)) = (
        tf.nn.bidirectional_dynamic_rnn(cell_fw=enc_cell,
                                    cell_bw=enc_cell,
                                    inputs=enc_embed_input,
                                    sequence_length=source_sequence_length,
                                    dtype=tf.float32, time_major=False)
    )

    

    enc_output = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)

    encoder_final_state_c = tf.concat(
    (encoder_fw_final_state.c, encoder_bw_final_state.c), 1)

    encoder_final_state_h = tf.concat(
        (encoder_fw_final_state.h, encoder_bw_final_state.h), 1)

    enc_state = LSTMStateTuple(
        c=encoder_final_state_c,
        h=encoder_final_state_h)
    
    return enc_output, enc_state

# Process the input we'll feed to the decoder
def process_decoder_input(target_data, vocab_to_int, batch_size):
    '''Remove the last word id from each batch and concat the <GO> to the begining of each batch'''
    ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    dec_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<GO>']), ending], 1)

    return dec_input

def decoding_layer(target_letter_to_int, decoding_embedding_size, num_layers, rnn_size,
                   target_sequence_length, max_target_sequence_length, enc_state, dec_input,enc_output,source_sequence_length):
    # 1. Decoder Embedding
    target_vocab_size = len(target_letter_to_int)
    dec_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoding_embedding_size]))
    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)

    # 2. Construct the decoder cell
    def make_cell(rnn_size):
        dec_cell = tf.contrib.rnn.LSTMCell(rnn_size,
                                           initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        return dec_cell


    if beam_width > 0 :
        
        enc_output=tf.contrib.seq2seq.tile_batch( enc_output, multiplier=beam_width )
        source_sequence_length = tf.contrib.seq2seq.tile_batch( source_sequence_length, multiplier=beam_width)
        enc_state = tf.contrib.seq2seq.tile_batch( enc_state, multiplier=beam_width )


    dec_cell=make_cell(rnn_size)
    attention_mechanism = tf.contrib.seq2seq.LuongAttention(
    rnn_size, enc_output,
    memory_sequence_length=source_sequence_length)
    
    dec_cell = tf.contrib.seq2seq.AttentionWrapper(
    dec_cell, attention_mechanism,
    attention_layer_size=rnn_size,alignment_history=True)

    # 3. Dense layer to translate the decoder's output at each time 
    # step into a choice from the target vocabulary
    output_layer = Dense(target_vocab_size,
                         kernel_initializer = tf.truncated_normal_initializer(mean = 0.0, stddev=0.1))


    # 4. Set up a training decoder and an inference decoder
    # Training Decoder
    with tf.variable_scope("decode"):

        # Helper for the training process. Used by BasicDecoder to read inputs.
        training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_embed_input,
                                                            sequence_length=target_sequence_length,
                                                            time_major=False)
        
        
        # Basic decoder
        training_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                           training_helper,
                                                           dec_cell.zero_state(batch_size,tf.float32).clone(cell_state=enc_state),
                                                           output_layer) 
        
        # Perform dynamic decoding using the decoder
        training_decoder_output , training_decoder_states , _ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                                       impute_finished=True,
                                                                       maximum_iterations=max_target_sequence_length)
    # 5. Inference Decoder
    # Reuses the same parameters trained by the training process
    with tf.variable_scope("decode", reuse=True):
        start_tokens = tf.tile(tf.constant([target_letter_to_int['<GO>']], dtype=tf.int32), [batch_size], name='start_tokens')

        # Helper for the inference process.
        if beam_width > 0:


            inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                cell=dec_cell,
                embedding=dec_embeddings,
                start_tokens=start_tokens,
                end_token=target_letter_to_int['<EOS>'],
                initial_state=dec_cell.zero_state(batch_size*beam_width,tf.float32).clone(cell_state=enc_state),
                beam_width=beam_width,
                output_layer=output_layer,
                length_penalty_weight=args.length_penalty_weight,
                )
            inference_decoder_output= tf.contrib.seq2seq.dynamic_decode(inference_decoder,
                                                                maximum_iterations=max_target_sequence_length)[0]

        else:
            inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(dec_embeddings,
                                                                    start_tokens,
                                                                    target_letter_to_int['<EOS>'])

            # Basic decoder
            inference_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                          inference_helper,
                                                          dec_cell.zero_state(batch_size,tf.float32).clone(cell_state=enc_state),
                                                          output_layer)
          
        # Perform dynamic decoding using the decoder
            inference_decoder_output,inference_decoder_states,_ = tf.contrib.seq2seq.dynamic_decode(inference_decoder,
                                                                maximum_iterations=max_target_sequence_length) 

    return training_decoder_output, training_decoder_states, inference_decoder_output

def seq2seq_model(input_data, targets, lr, target_sequence_length, 
                  max_target_sequence_length, source_sequence_length,
                  source_vocab_size, target_vocab_size,
                  enc_embedding_size, dec_embedding_size, 
                  rnn_size, num_layers):
    
    # Pass the input data through the encoder. We'll ignore the encoder output, but use the state
    enc_output, enc_state = encoding_layer(input_data, 
                                  rnn_size, 
                                  num_layers, 
                                  source_sequence_length,
                                  source_vocab_size, 
                                  encoding_embedding_size)
      
    # Prepare the target sequences we'll feed to the decoder in training mode
    dec_input = process_decoder_input(targets, target_letter_to_int, batch_size)
    
    # Pass encoder state and decoder inputs to the decoders
    training_decoder_output, training_decoder_states, inference_decoder_output = decoding_layer(target_letter_to_int, 
                                                                       decoding_embedding_size, 
                                                                       num_layers, 
                                                                       rnn_size,
                                                                       target_sequence_length,
                                                                       max_target_sequence_length,
                                                                       enc_state, 
                                                                       dec_input,enc_output,source_sequence_length) 
    
    return training_decoder_output, training_decoder_states, inference_decoder_output

def pad_sentence_batch(sentence_batch, pad_int):
    """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]

def get_batches(targets, sources, batch_size, source_pad_int, target_pad_int):
    """Batch targets, sources, and the lengths of their sentences together"""
    for batch_i in range(0, len(sources)//batch_size):
        start_i = batch_i * batch_size
        sources_batch = sources[start_i:start_i + batch_size]
        targets_batch = targets[start_i:start_i + batch_size]
        pad_sources_batch = np.array(pad_sentence_batch(sources_batch, source_pad_int))
        pad_targets_batch = np.array(pad_sentence_batch(targets_batch, target_pad_int))
        
        # Need the lengths for the _lengths parameters
        pad_targets_lengths = []
        for target in pad_targets_batch:
            pad_targets_lengths.append(len(target))
        
        pad_source_lengths = []
        for source in pad_sources_batch:
            pad_source_lengths.append(len(source))
        
        yield pad_targets_batch, pad_sources_batch, pad_targets_lengths, pad_source_lengths

"""## Train
We're now ready to train our model. If you run into OOM (out of memory) issues during training, try to decrease the batch_size.
"""
def source_to_seq(text):
    '''Prepare the text for the model'''
    sequence_length = len(text.split())
    return [source_letter_to_int.get(word) for word in text.split()]+ [source_letter_to_int['<PAD>']]*(sequence_length-len(text))
def plot_attention(attention_map, input_tags = None, output_tags = None):    
    attn_len = len(attention_map)

    # Plot the attention_map
    plt.clf()
    f = plt.figure(figsize=(30, 20))
    ax = f.add_subplot(1, 1, 1)

    # Add image
    i = ax.imshow(attention_map, interpolation='nearest', cmap='Blues')

    # Add colorbar
    cbaxes = f.add_axes([0.2, 0, 0.6, 0.03])
    cbar = f.colorbar(i, cax=cbaxes, orientation='horizontal')
    cbar.ax.set_xlabel('Alpha value (Probability output of the "softmax")', labelpad=2)
    len_otag=len(output_tags)
    # Add labels
    ax.set_yticks(range(len_otag))
    if output_tags != None:
      ax.set_yticklabels(output_tags[:len_otag])
    len_itag=len(input_tags)
    ax.set_xticks(range(len_itag))
    if input_tags != None:
      ax.set_xticklabels(input_tags[:len_itag], rotation=60)

    ax.set_xlabel('Input Sequence')
    ax.set_ylabel('Output Sequence')

    # add grid and legend
    ax.grid()
    plt.savefig("attention.png")

def fileout(path,source_sentences):
    checkpoint = args.save_dir+"/best_model"+str(25)+".ckpt"

    loaded_graph = tf.Graph()

    with tf.Session(graph=loaded_graph) as sess:
        # Load saved model
        loader = tf.train.import_meta_graph(checkpoint + '.meta')
        loader.restore(sess, checkpoint)
        i=0
        with open(path, 'w') as f:
            for sentences in source_sentences.split("\n"):
                input_sentence = sentences


                text = source_to_seq(input_sentence)



                input_data = loaded_graph.get_tensor_by_name('input:0')
                logits = loaded_graph.get_tensor_by_name('predictions:0')
                source_sequence_length = loaded_graph.get_tensor_by_name('source_sequence_length:0')
                target_sequence_length = loaded_graph.get_tensor_by_name('target_sequence_length:0')

                #Multiply by batch_size to match the model's input parameters
                answer_logits = sess.run(logits, {input_data: [text]*batch_size, 
                                                  target_sequence_length: [len(text)]*batch_size, 
                                                  source_sequence_length: [len(text)]*batch_size})[0] 


                pad = source_letter_to_int["<PAD>"]
                eos = source_letter_to_int["<EOS>"] 


                print('  Response Words: {}'.format(" ".join([target_int_to_letter[i] for i in np.squeeze(answer_logits).tolist() if i != pad and i!=eos])))
                f.write('{}\n'.format(" ".join([target_int_to_letter[i] for i in np.squeeze(answer_logits).tolist() if i != pad and i!=eos])))

# Split data to training and validation sets
def train(source_letter_ids,target_letter_ids,dev_source_letter_ids,dev_target_letter_ids,dev_sentences):
    train_source = source_letter_ids
    train_target = target_letter_ids
    valid_source = dev_source_letter_ids
    valid_target = dev_target_letter_ids


    display_step = 20 # Check training loss after every 20 batches
    avgval_loss=0
    prev_bleu=0

    with tf.Session(graph=train_graph) as sess:
        sess.run(tf.global_variables_initializer())
        avgtrain_loss=[]
        avgval_loss_list=[]    
        for epoch_i in range(1, epochs+1):
            train_loss_list=[]
            for batch_i, (targets_batch, sources_batch, targets_lengths, sources_lengths) in enumerate(
                    get_batches(train_target, train_source, batch_size,
                               source_letter_to_int['<PAD>'],
                               target_letter_to_int['<PAD>'])):
                
                # Training step
                _, loss , alignment= sess.run(
                    [train_op, cost ,training_decoder_states.alignment_history.stack()],
                    {input_data: sources_batch,
                     targets: targets_batch,
                     lr: learning_rate,
                     target_sequence_length: targets_lengths,
                     source_sequence_length: sources_lengths})
                # bp()
                # Debug message updating us on the status of the training
                train_loss_list.append(loss)
                if batch_i % display_step == 0 and batch_i > 0:
                    print('Epoch {:>3}/{} Batch {:>4}/{} - Loss: {:>6.3f}  '
                          .format(epoch_i,
                                  epochs, 
                                  batch_i, 
                                  len(train_source) // batch_size, 
                                  loss ))
                if  batch_i % display_step == 0 :
                    input_tags=[source_int_to_letter[p] for p in sources_batch[5]]
                    output_tags = [target_int_to_letter[p] for p in targets_batch[5]]
                    plot_attention(alignment[:,5,:], input_tags = input_tags, output_tags = output_tags)

            val_loss_list=[]
            for batch_j, (valid_targets_batch, valid_sources_batch, valid_targets_lengths, valid_sources_lengths) in enumerate(
                    get_batches(valid_target, valid_source, batch_size,
                           source_letter_to_int['<PAD>'],
                           target_letter_to_int['<PAD>'])):
                  # Calculate validation cost
                validation_loss = sess.run(
                    [cost],
                    {input_data: valid_sources_batch,
                    targets: valid_targets_batch,
                    lr: learning_rate,
                    target_sequence_length: valid_targets_lengths,
                    source_sequence_length: valid_sources_lengths})
                val_loss_list.append(validation_loss)

            avgval_loss=np.mean(np.array(val_loss_list))

            print('Epoch {:>3}/{} Validation loss: {:>6.3f} \n'
                      .format(epoch_i,
                              epochs, 
                              avgval_loss))
            with open("logging.txt", 'a') as logf:
                logf.write('Epoch {:>3}/{} Validation loss: {:>6.3f} \n'
                      .format(epoch_i,
                              epochs, 
                              avgval_loss))
            
            avgtrain_loss.append(np.mean(np.array(train_loss_list)))
            avgval_loss_list.append(avgval_loss)
            with open("logging.txt", 'a') as logf:
                logf.write("over epochs : Train average Loss List ==> {},Validation average Loss List ==> {} \n".format(avgtrain_loss,avgval_loss_list))
            
            if epoch_i>5 and avgval_loss_list[-1]>avgval_loss_list[-2]:
                args.earlystop -= 1
                if args.earlystop == 0 :
                    break
            if epoch_i%5==0 :
              fileout(args.save_dir+"/dev.txt",dev_sentences)
              new_bleu=final(args.save_dir+"/dev.txt","data/dev/summaries.txt",4)
              if prev_bleu > new_bleu:
                break
              prev_bleu = new_bleu

            if epoch_i%5==0:
                checkpoint = args.save_dir+"/best_model"+str(epoch_i)+".ckpt" 
                saver = tf.train.Saver()
                saver.save(sess, checkpoint)
                print('Model Trained and Saved')




##############################################################################################
##################  INPUT ############################################################

source_path = 'data/train/train.combined'
target_path = 'data/train/summaries.txt'
dev_source_path = 'data/dev/dev.combined'
dev_target_path = 'data/dev/summaries.txt'
test_source_path= 'data/test/test.combined'

source_sentences = helper.load_data(source_path)
target_sentences = helper.load_data(target_path)

dev_sentences = helper.load_data(dev_source_path)
dev_target_sentences = helper.load_data(dev_target_path)

test_sentences = helper.load_data(test_source_path)



# Build int2letter and letter2int dicts
source_int_to_letter, source_letter_to_int = extract_character_vocab(source_sentences)
target_int_to_letter, target_letter_to_int = extract_character_vocab(target_sentences)


# Convert characters to ids
source_letter_ids = [[source_letter_to_int.get(letter) for letter in line.split()] for line in source_sentences.split('\n')]
target_letter_ids = [[target_letter_to_int.get(letter) for letter in line.split()] + [target_letter_to_int['<EOS>']] for line in target_sentences.split('\n')] 

dev_source_letter_ids = [[source_letter_to_int.get(letter) for letter in line.split()] for line in dev_sentences.split('\n')]
dev_target_letter_ids = [[target_letter_to_int.get(letter) for letter in line.split()] + [target_letter_to_int['<EOS>']] for line in dev_target_sentences.split('\n')] 

"""################################# Hyperparameters #########################"""

# Number of Epochs
epochs = 30
# Batch Size
batch_size = args.batch_size
# RNN Size
rnn_size = 512
# Number of Layers
num_layers = 2
# Embedding Size
encoding_embedding_size = 256
decoding_embedding_size = 256
# Learning Rate
learning_rate = args.lr

keep_prob = 1.0 - args.use_dropout
keep_prob_placeholder =tf.placeholder(tf.float32, shape=[], name='keep_prob')

beam_width=args.beam_width

######################## Build the graph  ##################################################
train_graph = tf.Graph()
# Set the graph to default to ensure that it is ready for training
with train_graph.as_default():
    
    # Load the model inputs    
    input_data, targets, lr, target_sequence_length, max_target_sequence_length, source_sequence_length = get_model_inputs()
    
    # Create the training and inference logits
    training_decoder_output,training_decoder_states, inference_decoder_output = seq2seq_model(input_data, 
                                                                      targets, 
                                                                      lr, 
                                                                      target_sequence_length, 
                                                                      max_target_sequence_length, 
                                                                      source_sequence_length,
                                                                      len(source_letter_to_int),
                                                                      len(target_letter_to_int),
                                                                      encoding_embedding_size, 
                                                                      decoding_embedding_size, 
                                                                      rnn_size, 
                                                                      num_layers)    
    
    # Create tensors for the training logits and inference logits
    training_logits = tf.identity(training_decoder_output.rnn_output, 'logits')
    if beam_width > 0 :
        inference_logits = tf.identity(inference_decoder_output.predicted_ids, name='predictions')
    else:
        inference_logits = tf.identity(inference_decoder_output.sample_id, name='predictions')

    
    # Create the weights for sequence_loss
    masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name='masks')

    with tf.name_scope("optimization"):
        
        # Loss function
        cost = tf.contrib.seq2seq.sequence_loss(
            training_logits,
            targets,
            masks)

        # Optimizer
        optimizer = tf.train.AdamOptimizer(lr)

        # Gradient Clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)


# input_tags - word representation of input sequence, use None to skip
# output_tags - word representation of output sequence, use None to skip
# i - index of input element in batch
train(source_letter_ids,target_letter_ids,dev_source_letter_ids,dev_target_letter_ids,dev_sentences)
fileout(args.save_dir+"/dev.txt",dev_sentences)
fileout(args.save_dir+"/test_sumgen.txt",test_sentences)
print('Epoch {:>3} bleu score {:>3.4f}'.format(epochs,final(args.save_dir+"/dev.txt", 'data/summaries.txt',4)))