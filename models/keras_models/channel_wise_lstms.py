# +-------------------------------------------------------------------------------------------------+
# | channel_wise_lstms.py: model network architecture                                               |
# |                                                                                                 |
# | Eliane Röösli (2020), adapted from Harutyunyan et al (2019)                                     |
# +-------------------------------------------------------------------------------------------------+

from keras.models import Model
from keras.layers import Input, Dense, LSTM, Masking, Dropout
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.merge import Concatenate

from models.keras_utils import Slice, LastTimestep, ExtendMask

### BIDIRECTIONAL:
# In problems where all timesteps of the input sequence are available, Bidirectional LSTMs 
# train two instead of one LSTMs on the input sequence. The first on the input sequence as-is 
# and the second on a reversed copy of the input sequence. This can provide additional context 
# to the network and result in faster and even fuller learning on the problem.

### Recurrent dropout:
# masks connections between the recurrent units


### for further info: https://keras.io/layers/writing-your-own-keras-layers/

class Network(Model):

    def __init__(self, dim, batch_norm, dropout, rec_dropout, header, task, mask_demographics,
                 target_repl=False, deep_supervision=False, num_classes=1,
                 depth=1, input_dim=94, size_coef=4, **kwargs):

        self.dim = dim
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.rec_dropout = rec_dropout
        self.depth = depth
        self.size_coef = size_coef

        # (0) demographics: adjust input dimension and record retained variables
        included = ['GEN', 'ETH', 'INS']
        for dem in mask_demographics:
            if dem == 'Gender':
                input_dim -= 5
                included.remove('GEN')
            elif dem == 'Ethnicity':
                input_dim -= 6
                included.remove('ETH')
            elif dem == 'Insurance':
                input_dim -= 7
                included.remove('INS')
        if len(included) == 0:
            included.append("NONE")
        self._included = included
        
        # (1) define task-specific final activation layer
        if task in ['decomp', 'ihm', 'ph']:
            final_activation = 'sigmoid'
        elif task in ['los']:
            if num_classes == 1:
                final_activation = 'relu'
            else:
                final_activation = 'softmax'
        else:
            raise ValueError("Wrong value for task")

        print("==> not used params in network class:", kwargs.keys())

        
        # (2) Parse channels
        channel_names = set()
        
        # find: returns lowest index in string where substring is found
        # step necessary to clean up header after doing one-hot encoding
        for ch in header:
            # (a) not include if "mask->" is found 
            if ch.find("mask->") != -1:
                continue
            pos = ch.find("->")
            # (b) add header up to "->"
            if pos != -1:
                channel_names.add(ch[:pos])
            # (c) add full header
            else:
                channel_names.add(ch)
                
        channel_names = sorted(list(channel_names))
        self.channel_names = channel_names
        print("==> excluded demographics:", mask_demographics)
        print("==> found {} channels: {}".format(len(channel_names), channel_names))

        # each channel is a list of columns
        # step: select all channels associated with a certain header name (due to one-hot encoding)
        channels = [] 
        for ch in channel_names:
            indices = range(len(header))
            # only keep indices that correspond to retained channel names from header
            indices = list(filter(lambda i: header[i].find(ch) != -1, indices))
            channels.append(indices)
            
        # (3) Input layers and masking
        X = Input(shape=(None, input_dim), name='X')
        inputs = [X]
        mX = Masking()(X) # Masks a sequence by using a mask value to skip timesteps

        # (4) Deep supervision and bidirectionality
        if deep_supervision:
            M = Input(shape=(None,), name='M')
            inputs.append(M)

        is_bidirectional = True
        if deep_supervision:
            is_bidirectional = False

        # (5) Preprocess each channel
        cX = []
        for ch in channels:
            cX.append(Slice(ch)(mX)) # Slice 3D tensor by taking mX[:, :, ch]
        pX = []  # LSTM processed version of cX
        for x in cX:
            p = x
            for i in range(depth):
                num_units = dim
                if is_bidirectional:
                    num_units = num_units // 2

                lstm = LSTM(units=num_units,
                            activation='tanh',
                            return_sequences=True,
                            dropout=dropout,
                            recurrent_dropout=rec_dropout) 

                if is_bidirectional:
                    p = Bidirectional(lstm)(p)
                else:
                    p = lstm(p)
            pX.append(p)

        # (6) Concatenate processed channels
        Z = Concatenate(axis=2)(pX)

        # (7) Main part of the network
        for i in range(depth-1):
            num_units = int(size_coef*dim)
            if is_bidirectional:
                num_units = num_units // 2

            lstm = LSTM(units=num_units,
                        activation='tanh',
                        return_sequences=True,
                        dropout=dropout,
                        recurrent_dropout=rec_dropout)

            if is_bidirectional:
                Z = Bidirectional(lstm)(Z)
            else:
                Z = lstm(Z)

        # (8) Output module of the network
        return_sequences = (target_repl or deep_supervision)
        L = LSTM(units=int(size_coef*dim),
                 activation='tanh',
                 return_sequences=return_sequences,
                 dropout=dropout,
                 recurrent_dropout=rec_dropout)(Z)

        # (9) Additional tuning
        if dropout > 0:
            L = Dropout(dropout)(L)

        # (10) Output
        if target_repl:
            y = TimeDistributed(Dense(num_classes, activation=final_activation),
                                name='seq')(L)
            y_last = LastTimestep(name='single')(y)
            outputs = [y_last, y]
        elif deep_supervision:
            y = TimeDistributed(Dense(num_classes, activation=final_activation))(L)
            y = ExtendMask()([y, M])  # this way we extend mask of y to M
            outputs = [y]
        else:
            y = Dense(num_classes, activation=final_activation)(L)
            outputs = [y]

        # (11) build the specified network in keras
        super(Network, self).__init__(inputs=inputs, outputs=outputs)

        
        
        
    # print characteristics of network
    def say_name(self):
        return "{}.{}".format('k_clstms', "_".join(self._included))

    # print characteristics of network
    def say_name_old(self):
        return "{}.n{}.szc{}{}{}{}.dep{}".format('k_clstms',
                                                 self.dim,
                                                 self.size_coef,
                                                 ".bn" if self.batch_norm else "",
                                                 ".d{}".format(self.dropout) if self.dropout > 0 else "",
                                                 ".rd{}".format(self.rec_dropout) if self.rec_dropout > 0 else "",
                                                 self.depth)