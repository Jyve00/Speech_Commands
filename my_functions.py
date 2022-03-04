import torch 
import torchaudio 
import torch.nn as nn
import numpy as np 
import pandas as pd 




class TextTransform:
    """Maps characters to integers and vice versa"""
    def __init__(self):
        char_map_str = """
        ' 0
        <SPACE> 1
        a 2
        b 3
        c 4
        d 5
        e 6
        f 7
        g 8
        h 9
        i 10
        j 11
        k 12
        l 13
        m 14
        n 15
        o 16
        p 17
        q 18
        r 19
        s 20
        t 21
        u 22
        v 23
        w 24
        x 25
        y 26
        z 27
        """
        self.char_map = {}
        self.index_map = {}
        for line in char_map_str.strip().split('\n'):
            ch, index = line.split()
            self.char_map[ch] = int(index)
            self.index_map[int(index)] = ch
        self.index_map[1] = ' '

    def text_to_int(self, text):
        """ Use a character map and convert text to an integer sequence """
        int_sequence = []
        for c in text:
            if c == ' ':
                ch = self.char_map['<SPACE>']
            else:
                ch = self.char_map[c]
            int_sequence.append(ch)
        return int_sequence

    def int_to_text(self, labels):
        """ Use a character map and convert integer labels to an text sequence """
        string = []
        for i in labels:
            string.append(self.index_map[i])
        return ''.join(string).replace('<SPACE>', ' ')






    """Maps characters to integers and vice versa"""
    def __init__(self):
        char_map_str = """
        ' 0
        <SPACE> 1
        a 2
        b 3
        c 4
        d 5
        e 6
        f 7
        g 8
        h 9
        i 10
        j 11
        k 12
        l 13
        m 14
        n 15
        o 16
        p 17
        q 18
        r 19
        s 20
        t 21
        u 22
        v 23
        w 24
        x 25
        y 26
        z 27
        """
        self.char_map = {}
        self.index_map = {}
        for line in char_map_str.strip().split('\n'):
            ch, index = line.split()
            self.char_map[ch] = int(index)
            self.index_map[int(index)] = ch 
        self.index_map[1] = ' '

    def text_to_int(self, text):
        """ Use a character map and convert text to an integer sequence"""
        int_sequence = []
        for c in text:
            if c == ' ':
                ch = self.char_map['<SPACE>']
            else:
                ch = self.char_map[c]
            int_sequence.append(ch)
        return int_sequence

    def int_to_text(self, labels):
        """ Use a character map and convert integer labels to a text sequence"""
        string = []
        for i in labels:
            string.append(self.index_map[i])
        return ''.join(string).replace('<SPACE>', ' ')



#### DeepSpeech 
## link: https://pytorch.org/audio/stable/_modules/torchaudio/models/deepspeech.html#DeepSpeech





__all__ = ["DeepSpeech"]


class FullyConnected(torch.nn.Module):
    """
    Args:
        n_feature: Number of input features
        n_hidden: Internal hidden unit size.
    """

    def __init__(self,
                 n_feature: int,
                 n_hidden: int,
                 dropout: float,
                 relu_max_clip: int = 20) -> None:
        super(FullyConnected, self).__init__()
        self.fc = torch.nn.Linear(n_feature, n_hidden, bias=True)
        self.relu_max_clip = relu_max_clip
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.hardtanh(x, 0, self.relu_max_clip)
        if self.dropout:
            x = torch.nn.functional.dropout(x, self.dropout, self.training)
        return x



class DeepSpeech(nn.Module):
    """
    DeepSpeech model architecture from *Deep Speech: Scaling up end-to-end speech recognition* 
    [:footcite: 'hannun2014deep'].

    Args:
        n_features: Number of input features
        n_hidden: Internal hidden unit size.
        n_class: Number of output classes
    """

    def __init__(
        self, 
        n_feature: int, 
        n_hidden: int = 2048, 
        n_class: int = 40, 
        dropout: float = 0.0
    ) -> None:
        super(DeepSpeech, self).__init__()
        self.N_hidden = n_hidden
        self.fc1 = FullyConnected(n_feature, n_hidden, dropout)
        self.fc2 = FullyConnected(n_feature, n_hidden, dropout)
        self.fc3 = FullyConnected(n_feature, n_hidden, dropout)
        self.bi_rnn = nn.RNN(
            n_hidden, n_hidden, num_layers=1, nonlinearity="relu", bidirectional=True
        )
        self.fc4 = FullyConnected(n_hidden, n_hidden, dropout)
        self.out = nn.Linear(n_hidden, n_class)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Args:
                x (torch.Tensor): Tensor of dimension (batch, channel, time, feature).
            Returns:
                Tensor: Predictor tensor of dimension (batch, time, class).
            """
            # N x C x T x F
            x = self.fc1(x)
            # N x C x T x F
            x = self.fc2(x)
            # N x C x T x F
            x = self.fc3(x)
            # N x C x T x F
            x = x.squeeze(1)
            # N x T x H 
            x = x.transpose(0, 1)
            # T x N x H 
            x, _ = self.bi_rnn(x)
            # The fifth (non-recurrent) layer takes both the forward and backward units as inputs 
            x = x[:, :, :self.n_hidden] + x[:, :, self.n_hidden:]
            # T x N xH 
            x = self.fc4(x)
            # T x N x H
            x = self.out(x)
            # T x N x n_class 
            x = x.permute(1, 0, 2)
            # N x T x n_class 
            x = nn.functional.log_softmax(x, dim=2)
            # N x T x n_class 
            return x 



def GreedyDecoder(output, labels, label_lengths, blank_label=28, collapse_repeated=True):
	"""outputs the most probable character at each time step.

	Args:
		output (_type_): _description_
		labels (_type_): _description_
		label_lengths (_type_): _description_
		blank_label (int, optional): _description_. Defaults to 28.
		collapse_repeated (bool, optional): _description_. Defaults to True.

	Returns:
		_type_: _description_
	"""
	arg_maxes = torch.argmax(output, dim=2)
	decodes = []
	targets = []
	for i, args in enumerate(arg_maxes):
		decode = []
		targets.append(text_transform.int_to_text(labels[i][:label_lengths[i]].tolist()))
		for j, index in enumerate(args):
			if index != blank_label:
				if collapse_repeated and j != 0 and index == args[j -1]:
					continue
				decode.append(index.item())
		decodes.append(text_transform.int_to_text(decode))
	return decodes, targets