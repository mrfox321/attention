import tensorflow as tf
import numpy as np
from model import *

dim_model = 512
num_heads = 8
batch_size = 16
num_layers = 5
seq_len = 100
max_len = 20
vocab_size = 1000
start_token = tf.constant(0)
end_token = tf.constant(1)
pad_token = tf.constant(2)
embedding = tf.random_uniform(shape=([vocab_size,dim_model]))

sequence = tf.placeholder(tf.float32,shape=(batch_size,seq_len,dim_model))

in_seq = tf.placeholder(tf.int32,shape=(batch_size,seq_len))
out_seq = tf.placeholder(tf.int32,shape=(batch_size,seq_len-30))

with tf.variable_scope("attention_test"):
    attention = Attention(num_heads,dim_model)
    for bool in [False,True]:
        attention.attend(sequence, sequence, sequence, head=0, use_mask=bool)
        attention.multi_attend(sequence, sequence, sequence, use_mask=bool)

with tf.variable_scope("encodinglayer_test"):
    encodinglayer = EncodingLayer(num_heads,dim_model)
    encodinglayer.forward(sequence)

with tf.variable_scope("encoder_test"):
    encoder = Encoder(num_layers,num_heads,dim_model)
    encoding = encoder.encode(sequence)

with tf.variable_scope("decodinglayer_test"):
    decodinglayer = DecodingLayer(num_heads,dim_model)
    decodinglayer.forward(sequence,encoding)
    decodinglayer.forward_inference(sequence,encoding)

with tf.variable_scope("decoder_test"):
    decoder = Decoder(num_layers,num_heads,dim_model)
    decoding = decoder.decode(sequence,encoding)

with tf.variable_scope("greedy_decode_test"):
    greedydecoder = GreedyDecoder(embedding,num_layers,num_heads,dim_model,max_len,
                                  start_token,end_token,pad_token)
    greedydecoder.add_encoding(encoding)
    layer_state = tf.ones(shape=(num_layers+1,tf.shape(encoding)[0],1,tf.shape(encoding)[-1]),dtype=tf.float32)
    sequence = tf.random_uniform(shape=(16,tf.shape(encoding)[-1]))
    output = greedydecoder.decode(sequence,layer_state,0)

with tf.variable_scope("greedy_inference_test"):
    greedydecoder = GreedyDecoder(embedding,num_layers,num_heads,dim_model,max_len,
                                  start_token,end_token,pad_token)
    greedydecoder.add_encoding(encoding)
    layer_state = tf.ones(shape=(num_layers+1,tf.shape(encoding)[0],1,tf.shape(encoding)[-1]),dtype=tf.float32)
    input_tokens = tf.ones(shape=(16,),dtype=tf.int32)
    output_tokens, layer_state = greedydecoder.inference(input_tokens,layer_state,0)

with tf.variable_scope("greedy_decode_step_test"):
    greedydecoder = GreedyDecoder(embedding,num_layers,num_heads,dim_model,max_len,
                                  start_token,end_token,pad_token)
    greedydecoder.add_encoding(encoding)
    layer_state = tf.ones(shape=(num_layers+1,tf.shape(encoding)[0],1,tf.shape(encoding)[-1]),dtype=tf.float32)
    sequence = tf.ones(shape=(16,2),dtype=tf.int32)
    loop_index = tf.constant(2)
    sequence, layer_state, loop_index = greedydecoder.decode_step(sequence,layer_state,loop_index)

with tf.variable_scope("full_decode_test"):
    greedydecoder = GreedyDecoder(embedding,num_layers,num_heads,dim_model,max_len,
                                  start_token,end_token,pad_token)
    greedydecoder.add_encoding(encoding)
    output_sequence = greedydecoder.greedy_decode(encoding)

with tf.variable_scope("transformer_test"):
    transformer = Transformer(embedding,num_layers,num_heads,dim_model)
    scores = transformer.transform(in_seq,out_seq)

with tf.variable_scope("greedy_transformer_test"):
    greedytransformer = GreedyTransformer(embedding,num_layers,num_heads,dim_model,max_len,start_token,end_token,pad_token)
    greedy_output = greedytransformer.transform(in_seq)
