import tensorflow as tf
import numpy as np

def mask(x):
    """
    x[batch,time_query,time_key] : time_query = time_key
    """
    seq_lens = tf.range(1,tf.shape(x)[1]+1)
    mask_tensor = tf.sequence_mask(seq_lens,dtype=tf.float32)*tf.float32.max - tf.float32.max
    return x*mask_tensor

class Attention(object):

    def __init__(self,num_heads,dim_model):

        self.dim_model = dim_model
        self.num_heads = num_heads
        self.dim_key = dim_model / num_heads
        self.dim_value = dim_model / num_heads

        self.key_weights = [tf.get_variable("key_weights_"+str(i),
                                       shape=[self.dim_model,self.dim_key],
                                       dtype=tf.float32,
                                       trainable=True) for i in xrange(num_heads)]
        self.value_weights = [tf.get_variable("value_weights_"+str(i),
                                       shape=[self.dim_model,self.dim_value],
                                       dtype=tf.float32,
                                       trainable=True) for i in xrange(num_heads)]
        self.query_weights = [tf.get_variable("query_weights_"+str(i),
                                       shape=[self.dim_model,self.dim_key],
                                       dtype=tf.float32,
                                       trainable=True) for i in xrange(num_heads)]
        self.multi_weights = tf.get_variable("multi_weights",
                                             shape=[self.num_heads*self.dim_value,self.dim_model],
                                             dtype=tf.float32,
                                             trainable=True)

    def attend(self,key,value,query,head,use_mask):

        #can also expand and tile then tf.matmul instead of einsum
        lin_key = tf.einsum('ijk,km->ijm',key,self.key_weights[head])
        lin_value = tf.einsum('ijk,km->ijm',value,self.value_weights[head])
        lin_query = tf.einsum('ijk,km->ijm',query,self.query_weights[head])

        lin_key = tf.transpose(lin_key, perm=[0,2,1])

        query_key = tf.matmul(lin_query,lin_key) / tf.sqrt(tf.cast(self.dim_key,tf.float32))
        if use_mask:
            query_key = mask(query_key)

        alignments = tf.nn.softmax(query_key)

        return tf.matmul(alignments,lin_value)

    def multi_attend(self,key,value,query,use_mask):

        heads = []
        for head in xrange(self.num_heads):
            heads.append(self.attend(key,value,query,head,use_mask))
        multi_head = tf.concat(heads,axis=-1)
        return tf.einsum('ijk,km->ijm',multi_head,self.multi_weights)

def dense_norm(x):
    """
    x[batch,time,dim]
    """
    y = tf.layers.dense(x,units=2048,activation=tf.nn.relu,name='dense_1',reuse=tf.AUTO_REUSE)
    y = tf.layers.dense(y,units=512,name='dense_2',reuse=tf.AUTO_REUSE)
    return tf.contrib.layers.layer_norm(x + y, reuse=tf.AUTO_REUSE, scope='last_norm')

class EncodingLayer(object):

    def __init__(self,num_heads,dim_model):

        self.attention_sublayer = Attention(num_heads,dim_model)

    def forward(self,x):

        y = self.attention_sublayer.multi_attend(key=x,value=x,query=x,use_mask=False)
        x = tf.contrib.layers.layer_norm(x + y, reuse=tf.AUTO_REUSE, scope='norm_1')
        return dense_norm(x)

class Encoder(object):

    def __init__(self,num_layers,num_heads,dim_model):

        self.layers = []
        for i in xrange(num_layers):
            with tf.variable_scope("EncodingLayer_" + str(i), reuse=tf.AUTO_REUSE):
                self.layers.append(EncodingLayer(num_heads,dim_model))

    def encode(self,sequence):
        """
        sequence[batch,time,dim]
        """
        encoding = sequence
        for i,layer in enumerate(self.layers):
            with tf.variable_scope("Encode_" + str(i), reuse=tf.AUTO_REUSE):
                encoding = layer.forward(encoding)
        return encoding

class DecodingLayer(object):

    def __init__(self,num_heads,dim_model):

        with tf.variable_scope("selfAttention"):
            self.self_attention_sublayer = Attention(num_heads,dim_model)
        with tf.variable_scope("mixedAttention"):
            self.mixed_attention_sublayer = Attention(num_heads,dim_model)


    def forward(self,x,encoding):

        #self-attention sublayer
        y = self.self_attention_sublayer.multi_attend(key=x,value=x,query=x,use_mask=True)
        x = tf.contrib.layers.layer_norm(x + y, reuse=tf.AUTO_REUSE, scope='norm_1')

        #mixed-attention sublayer
        y = self.mixed_attention_sublayer.multi_attend(key=encoding,value=encoding,query=x,use_mask=False)
        x = tf.contrib.layers.layer_norm(x + y, reuse=tf.AUTO_REUSE, scope='norm_2')
        return dense_norm(x)

    def forward_inference(self,x,encoding):

        #auto-regressive model => only need to use last element of sequence as query
        q = tf.expand_dims(x[:,-1],axis=1)

        #self-attention sublayer => need previous elements of decoded sequence for keys and values
        y = self.self_attention_sublayer.multi_attend(key=x,value=x,query=q,use_mask=False)
        q = tf.contrib.layers.layer_norm(q + y, reuse=tf.AUTO_REUSE, scope='norm_1')

        #mixed-attention sublayer
        y = self.mixed_attention_sublayer.multi_attend(key=encoding,value=encoding,query=q,use_mask=False)
        q = tf.contrib.layers.layer_norm(q + y, reuse=tf.AUTO_REUSE, scope='norm_2')
        return dense_norm(q)


class Decoder(object):

    def __init__(self,num_layers,num_heads,dim_model):

        self.layers = []
        for i in xrange(num_layers):
            with tf.variable_scope("DecodingLayer_" + str(i)):
                self.layers.append(DecodingLayer(num_heads,dim_model))

    def decode(self,sequence,encoding):
        """
        sequence[batch,time,dim]
        """
        decoding = sequence
        for i, layer in enumerate(self.layers):
            with tf.variable_scope("Decode_" + str(i)):
                decoding = layer.forward(decoding,encoding)
        return decoding

class EncoderDecoder(object):

    def __init__(self,num_layers,num_heads,dim_model):

        #construct objects representing neural network
        self.encoder = Encoder(num_layers,num_heads,dim_model)
        self.decoder = Decoder(num_layers,num_heads,dim_model)

    def transduce(self,input_sequence,output_sequence):

        encoding = self.encoder.encode(input_sequence)
        decoding = self.decoder.decode(output_sequence,encoding)
        return decoding

def pos_encode(sequence):
    """
    Full sequence position encoding
    sequence[batch,time,dim] -> full sequence position encoding
    """
    pos = tf.range(tf.shape(sequence)[1])
    return full_pos_encode(sequence,pos)

def pos_encode_time(sequence, time):
    """
    Encoding sequence at a given time
    sequence[batch,dim] -> single position encoding
    time : tf.float32
    """
    pos = tf.reshape(time,shape=(1,))
    sequence = tf.expand_dims(sequence,axis=1)
    return tf.squeeze(full_pos_encode(sequence,pos))

def full_pos_encode(sequence, pos):
    """
    sequence[batch,time,dim]
    pos[time] : tf.int32
    """
    #pos[time] -> pos[1,time] : tf.float32
    pos = tf.expand_dims(pos,axis=0)
    pos = tf.cast(pos,tf.float32)

    #exponent[dim/2] -> exponent[dim/2,1]
    exponent = tf.range(start=2,limit=tf.shape(sequence)[-1]+1,delta=2)
    exponent = tf.cast(exponent,tf.float32)
    exponent = exponent / tf.cast(tf.shape(sequence)[-1],tf.float32)
    exponent = tf.expand_dims(exponent,axis=-1)

    #theta[dim/2,time]
    theta = pos / 10000**exponent

    even = tf.sin(theta)
    odd = tf.cos(theta)

    #encode[dim/2,2*time]
    encode = tf.concat([even,odd],axis=-1)
    #encode[dim/2,2*time] -> encode[dim,time]
    encode = tf.reshape(encode,shape=(tf.shape(sequence)[-1],tf.shape(sequence)[1]))
    #encode[dim,time] -> encode[time,dim]
    encode = tf.transpose(encode)
    return sequence + encode

def transduce(input_sequence,output_sequence,embedding,params):
    """
    sequence[batch,time] : tf.int32 (integer token representations of words)
    embedding[word,dim]  : tf.float32 (word embedding)
    """
    input_sequence = tf.gather(embedding,input_sequence)
    output_sequence = tf.gather(embedding,output_sequence)
    input_sequence, output_sequence = pos_encode(input_sequence), pos_encode(output_sequence)

    transformer = EncoderDecoder(params.num_layers,params.num_heads,params.dim_model)
    decoding = transformer.transduce(input_sequence,output_sequence)

    scores = tf.einsum('ijk,mk->ijm',decoding,embedding)
    return scores

def get_embedding(initial_embedding,vocab_size,hidden_dim):
    """
    initial_embedding[vocab_size,hidden_dim] : np.float32 (numpy array)
    """
    init = tf.constant_initializer(initial_embedding)
    embedding = tf.get_variable('embedding',
                                shape=[vocab_size,hidden_dim],
                                dtype=tf.float32,
                                trainable=True,
                                initializer=init)
    return embedding

def layer_concat(layer_state,decoding):
    """
    layer_state[layer,batch,time,dim]
    decoding[batch,1,dim]
    """
    decoding = tf.expand_dims(decoding,axis=0)
    return tf.concat([layer_state,decoding],axis=0)


class GreedyDecoder(object):

    def __init__(self,num_layers,num_heads,dim_model,max_len,start_token,end_token,pad_token):

        self.max_len = max_len
        self.start_token = start_token
        self.end_token = end_token
        self.pad_token = pad_token
        self.layers = []
        for i in xrange(num_layers):
            with tf.variable_scope("DecodingLayer_" + str(i)):
                self.layers.append(DecodingLayer(num_heads,dim_model))

    def add_encoding(self,encoding):
        self.encoding = encoding

    def add_embedding(self,embedding):
        self.embedding = embedding

    def first_pass(self,layer_state):
        """
        layer_state[1,batch,1,dim] -> layer_state[layer,batch,1,dim]
        """
        for i, layer in enumerate(self.layers):
            with tf.variable_scope("Decode_" + str(i)):
                decoding = layer.forward_inference(layer_state[i],self.encoding)
                layer_state = layer_concat(layer_state,decoding)
        return layer_state

    def next_pass(self,layer_state,next_sequence):
        """
        layer_state[layer,batch,time,dim] : initializes as layer_state[layer,batch,1,dim]
        layer_state[-1] is output of stacked decoder layers
        next_sequence[batch,1,dim]
        """
        #decodings[1,batch,1,dim]
        decodings = tf.expand_dims(next_sequence,axis=0)
        for i, layer in enumerate(self.layers):
            with tf.variable_scope("DecodingLayer_" + str(i)):
                #concat along time
                #input_sequence[batch,time,dim] -> input_sequence[batch,time+1,dim]
                input_sequence = tf.concat([layer_state[i],decodings[-1]],axis=1)
                #decoding[batch,1,dim] -> decoding[1,batch,1,dim]
                decoding = tf.expand_dims(layer.forward_inference(input_sequence,self.encoding),axis=0)
                #decodings[layer,batch,1,dim] -> decodings[layer+1,batch,1,dim]
                decodings = tf.concat([decodings,decoding],axis=0)
        layer_state = tf.concat([layer_state,decodings],axis=2)
        return layer_state

    def decode_loop(self,layer_state,next_sequence,loop_index):
        return tf.cond(tf.equal(loop_index,0),
                       lambda : self.first_pass(layer_state),
                       lambda : self.next_pass(layer_state,next_sequence))

    def decode_loop_1(self,layer_state,next_sequence,loop_index):
        """
        layer_state[layer,batch,time,dim] : tf.float32
        when loop_index == 0:
        layer_state is a dummy variable with shape layer_state[layer,batch,1,dim]

        next_sequence[batch,dim] : tf.float32
        loop_index : tf.int32
        """
        # First iteration has boundary conditions
        # since layer_state is an unused dummy variable
        # In subsequent iterations layer_state builds as
        # layer_state[layer,batch,time,dim] -> layer_state[layer,batch,time+1,dim]
        predicate = tf.equal(loop_index,0)
        decodings = tf.expand_dims(tf.expand_dims(next_sequence,axis=1),axis=0)
        for i, layer in enumerate(self.layers):
            with tf.variable_scope("Decode_" + str(i)):
                #layer_state[layer,batch,1,dim] is not used in first loop of decode step
                input_sequence = tf.cond(
                   predicate,
                   lambda : decodings[-1],
                   lambda : tf.concat([layer_state[i],decodings[-1]],axis=1))
                #decoding[batch,1,dim] -> decoding[1,batch,1,dim]
                decoding = tf.expand_dims(layer.forward_inference(input_sequence,self.encoding),axis=0)
                decodings = tf.concat([decodings,decoding],axis=0)

        layer_state = tf.cond(
           predicate,
           lambda : decodings,
           lambda : tf.concat([layer_state,decodings],axis=2))
        return layer_state

    def inference(self,input_tokens,layer_state,loop_index):
        """
        inputs:
        input_tokens[batch] : tf.int32

        layer_state[layer,batch,time,dim] : tf.float32
        when loop_index == 0:
        layer_state is a dummy variable with shape:
        layer_state[layer,batch,1,dim]

        loop_index : tf.int32

        outputs:
        output_tokens[batch] : tf.int32
        layer_state[layer,batch,time+1,dim] : tf.float32
        """
        input_sequence = tf.gather(self.embedding,input_tokens)
        input_sequence = pos_encode_time(input_sequence,loop_index)
        #layer_state[layer,batch,time,dim] -> layer_state[layer,batch,time+1,dim]
        #layer_state[:,:,-1,:] is computed by self.decode and concatenated
        layer_state = self.decode_loop(layer_state,input_sequence,loop_index)
        last_sequence = layer_state[-1,:,-1,:]
        last_sequence = tf.matmul(last_sequence,tf.transpose(self.embedding))
        output_tokens = tf.argmax(last_sequence,axis=-1,output_type=tf.int32)
        return output_tokens, layer_state

    def end_condition(self,sequence):
        """
        sequence[batch,time] : tf.int32
        max_len : tf.int32
        end_token : tf.int32
        pad_token : tf.int32
        """
        last_tokens = sequence[:,-1]
        end_token_condition = tf.equal(last_tokens,self.end_token)
        pad_token_condition = tf.equal(last_tokens,self.pad_token)
        end_decode_condition = tf.logical_or(end_token_condition, pad_token_condition)
        end_decode_condition = tf.reduce_all(end_decode_condition)
        return end_decode_condition

    def body(self,sequence,layer_state,loop_index):
        """
        sequence[batch,time] : tf.int32
        layer_state[layer,batch,time,dim]
        initialied as sequence[batch,1] = start_token
        """
        output_tokens, layer_state = self.inference(sequence[:,-1],layer_state,loop_index)
        sequence = tf.concat([sequence,tf.expand_dims(output_tokens,axis=-1)],axis=-1)
        return sequence, layer_state, loop_index + 1

    def full_decode(self,encoding,embedding):

        self.add_encoding(encoding)
        self.add_embedding(embedding)
        #sequence[batch,1] : tf.int32
        sequence = self.start_token*tf.ones(shape=(tf.shape(encoding)[0],1),dtype=tf.int32)
        #layer_state initialized as a dummy variable with shape
        #layer_state[layer,batch,1,dim] for the first iteration in tf.while
        dummy_shape = (len(self.layers)+1,tf.shape(encoding)[0],1,tf.shape(encoding)[-1])
        layer_state = tf.ones(shape=dummy_shape,dtype=tf.float32)
        loop_index = tf.constant(0,dtype=tf.int32)

        while_func = tf.while_loop(lambda sequence,layer_state,loop_index : self.end_condition(sequence),
                                   self.body,
                                   (sequence, layer_state, loop_index),
                                   maximum_iterations=self.max_len)

        #while_func = (input_tokens, layer_state, loop_index)
        #layer_state[-1] is last layer (output) of network
        output_sequence = while_func[1][-1]
        return output_sequence
