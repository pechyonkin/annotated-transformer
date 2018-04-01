
# The Annotated "Attention is All You Need"

> ### By Alexander Rush (@harvardnlp)



> When I teach machine learning, I often tell students that it is impossible to understand recent papers without getting their hands dirty and jumping in. While ideally math should communicate what exactly is going on, many of the "details" turn out to be quite important to getting good results.

> The following is an exercise in practicing what I preach. The recent ["Attention is All You Need"/Transformer](https://arxiv.org/abs/1706.03762) paper from NIPS 2017 has been an instantly impactful paper in machine translation and likely NLP generally. The paper reads in a very clean way, but the conventional wisdom has been that it is very difficult to implement correctly. 

> This blog post presents an annotated guide to the paper. My goal is to implement every detail from the paper (and its references) inline with the paper itself. To do so, I will interleave the text from the paper, with a line-by-line PyTorch re-implementation. (I have done a bit of reordering of the original paper, and skip some of the non-implementation sections for brevity).

> This document itself is a working notebook, that should be usable as a Transformer implementation. To run it first install [PyTorch](http://pytorch.org/) and [torchtext](https://github.com/pytorch/text). 



```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext import data, datasets
import math, copy
from torch.autograd import Variable
```

# Abstract 
The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being  more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU.  On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing  both with large and limited training data.

<img src="ModalNet-21.png" width=400px>

# Model

Most competitive neural sequence transduction models have an encoder-decoder structure [(cite)](cho2014learning,bahdanau2014neural,sutskever14). Here, the encoder maps an input sequence of symbol representations $(x_1, ..., x_n)$ to a sequence of continuous representations $\mathbf{z} = (z_1, ..., z_n)$. Given $\mathbf{z}$, the decoder then generates an output sequence $(y_1,...,y_m)$ of symbols one element at a time. At each step the model is auto-regressive [(cite)](graves2013generating), consuming the previously generated symbols as additional input when generating the next.                                                
                                                                                                                                                                                                                  
The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder, shown in the left and right halves of Figure~1, respectively. 

> Our setup to start will be a standard neural sequence model. We will assume the src and tgt sequences are passed through an encoder, converted to a memory, and passed through a decoder model.


```python
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator, pad_idx):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        self.pad_idx = pad_idx
        
    def forward(self, src, tgt):
        src_mask = (src != self.pad_idx).unsqueeze(-2)
        tgt_mask = (tgt != self.pad_idx).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        memory = self.encoder(self.src_embed(src), src_mask)
        output = self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
        return output

```

## Encoder and Decoder Stacks                                                                                                                                                                                
                                                                                                                                                                                                                                                                                                                     
### Encoder: 

The encoder is composed of a stack of $N=6$ identical layers. 


```python
def clones(module, N):
    "Produces N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
```


```python
class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        "Pass the input through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
```

We employ a residual connection [(cite)](he2016deep) around each of the two sub-layers, followed by layer normalization [(cite)](layernorm2016).  


```python
class LayerNorm(nn.Module):
    "Construct a layernorm module based on above citation."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
```

That is, the output of each sub-layer is $\mathrm{LayerNorm}(x + \mathrm{Sublayer}(x))$, where $\mathrm{Sublayer}(x)$ is the function implemented by the sub-layer itself.  We apply dropout \citep{srivastava2014dropout} to the output of each sub-layer, before it is added to the sub-layer input and normalized.   


```python
class SublayerConnection(nn.Module):
    "A residual connection followed by a layer norm."
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
```

To facilitate these residual connections, all sub-layers in the model, as well as the embedding layers, produce outputs of dimension $d_{\text{model}}=512$.   

Each layer has two sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, position-wise fully connected feed-forward network.


```python
class EncoderLayer(nn.Module):
    "Encoder is made up of two sublayers, self attn and feed forward."
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, mask=mask))
        return self.sublayer[1](x, self.feed_forward)
```

### Decoder:

The decoder is also composed of a stack of $N=6$ identical layers.  



```python
class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
```

In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack.  Similar to the encoder, we employ residual connections around each of the sub-layers, followed by layer normalization.  


```python
class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, mask=tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)
```

We also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions.  This masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position $i$ can depend only on the known outputs at positions less than $i$.


```python
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0
```

<img width="220px" src="ModalNet-19.png">

### Attention:                                                                                                                                                                                                                                                                               
An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors.  The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.                                                                                                             


#### Scaled Dot-Product Attention                                                                                                                                                                               

We call our particular attention "Scaled Dot-Product Attention" (Figure 2).   The input consists of queries and keys of dimension $d_k$, and values of dimension $d_v$.  We compute the dot products of the query with all keys, divide each by $\sqrt{d_k}$, and apply a softmax function to obtain the weights on the values.                                                                                                         

In practice, we compute the attention function on a set of queries simultaneously, packed together into a matrix $Q$.   The keys and values are also packed together into matrices $K$ and $V$.  We compute the matrix of outputs as:                      
                                                                 
$$                                                                         
   \mathrm{Attention}(Q, K, V) = \mathrm{softmax}(\frac{QK^T}{\sqrt{d_k}})V               
$$                                                                                                                                                                                                        
                                                                                                                                                                     


```python
def attention(query, key, value, mask=None, dropout=0.0):
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attn = F.softmax(scores, dim = -1)
    return torch.matmul(F.dropout(attn, p=dropout), value), attn
```

The two most commonly used attention functions are additive attention \citep{bahdanau2014neural}, and dot-product (multiplicative) attention.  Dot-product attention is identical to our algorithm, except for the scaling factor of $\frac{1}{\sqrt{d_k}}$. Additive attention computes the compatibility function using a feed-forward network with a single hidden layer.  While the two are similar in theoretical complexity, dot-product attention is much faster and more space-efficient in practice, since it can be implemented using highly optimized matrix multiplication code.                                                                                             

                                                                        
While for small values of $d_k$ the two mechanisms perform similarly, additive attention outperforms dot product attention without scaling for larger values of $d_k$ \citep{DBLP:journals/corr/BritzGLL17}. We suspect that for large values of $d_k$, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients  \footnote{To illustrate why the dot products get large, assume that the components of $q$ and $k$ are independent random variables with mean $0$ and variance $1$.  Then their dot product, $q \cdot k = \sum_{i=1}^{d_k} q_ik_i$, has mean $0$ and variance $d_k$.}. To counteract this effect, we scale the dot products by $\frac{1}{\sqrt{d_k}}$.          

 

### Multi-Head Attention                                                                                                                                                                                                                                                                                                                                                                                                                               
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
Instead of performing a single attention function with $d_{\text{model}}$-dimensional keys, values and queries, we found it beneficial to linearly project the queries, keys and values $h$ times with different, learned linear projections to $d_k$, $d_k$ and $d_v$ dimensions, respectively.                                                                                                                                                                                                   
On each of these projected versions of queries, keys and values we then perform the attention function in parallel, yielding $d_v$-dimensional output values. These are concatenated and once again projected, resulting in the final values:

<img width="270px" src="ModalNet-20.png">
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this.                                                                                                                                                                                                                                                                                             
    
    
   
$$    
\mathrm{MultiHead}(Q, K, V) = \mathrm{Concat}(\mathrm{head_1}, ..., \mathrm{head_h})W^O    \\                                           
    \text{where}~\mathrm{head_i} = \mathrm{Attention}(QW^Q_i, KW^K_i, VW^V_i)                                
$$                                                                                                                                                                                                                                                                                                                                                                         
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
Where the projections are parameter matrices $W^Q_i \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $W^K_i \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $W^V_i \in \mathbb{R}^{d_{\text{model}} \times d_v}$ and $W^O \in \mathbb{R}^{hd_v \times d_{\text{model}}}$.                                                                                                                                                                                                                                                        
   

   
In this work we employ $h=8$ parallel attention layers, or heads. For each of these we use $d_k=d_v=d_{\text{model}}/h=64$.                                                                                                                                                                                                                                                                                                                                                                        
Due to the reduced dimension of each head, the total computational cost is similar to that of single-head attention with full dimensionality.   


```python
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.p = dropout
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        
    def forward(self, query, key=None, value=None, mask=None):
        if key is None:
            key = query
            value = query
        if mask is not None:
            mask = mask.unsqueeze(1)
        batches = query.size(0)
        def shape(x):
            return x.view(batches, -1, self.h, self.d_k).transpose(1, 2)
        def unshape(x):
            return x.transpose(1, 2).contiguous().view(batches, -1, self.h * self.d_k)
        
        query, key, value = [shape(l(x)) 
                             for l, x in zip(self.linears, (query, key, value))]
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.p)
        x = unshape(x)
        return self.linears[-1](x)
```

### Applications of Attention in our Model                                                                                                                                                      
The Transformer uses multi-head attention in three different ways:                                                        
* In "encoder-decoder attention" layers, the queries come from the previous decoder layer, and the memory keys and values come from the output of the encoder.   This allows every position in the decoder to attend over all positions in the input sequence.  This mimics the typical encoder-decoder attention mechanisms in sequence-to-sequence models such as \citep{wu2016google, bahdanau2014neural,JonasFaceNet2017}.                                                        
* The encoder contains self-attention layers.  In a self-attention layer all of the keys, values and queries come from the same place, in this case, the output of the previous layer in the encoder.   Each position in the encoder can attend to all positions in the previous layer of the encoder.                                                                
* Similarly, self-attention layers in the decoder allow each position in the decoder to attend to all positions in the decoder up to and including that position.  We need to prevent leftward information flow in the decoder to preserve the auto-regressive property.  We implement this inside of scaled dot-product attention by masking out (setting to $-\infty$) all values in the input of the softmax which correspond to illegal connections.  See Figure~\ref{fig:multi-head-att}.                                                                                                                                                                                                                                                                                                                           

## Position-wise Feed-Forward Networks                                                                                                                                                                                                                                                                                                                                                             
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
In addition to attention sub-layers, each of the layers in our encoder and decoder contains a fully connected feed-forward network, which is applied to each position separately and identically.  This consists of two linear transformations with a ReLU activation in between.

$$\mathrm{FFN}(x)=\max(0, xW_1 + b_1) W_2 + b_2$$                                                                                                                                                                                                                                                         
                                                                                                                                                                                                                                                        
While the linear transformations are the same across different positions, they use different parameters from layer to layer. Another way of describing this is as two convolutions with kernel size 1.  The dimensionality of input and output is $d_{\text{model}}=512$, and the inner-layer has dimensionality $d_{ff}=2048$. 


```python
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
```

## Embeddings and Softmax                                                                                                                                                                                                                                                                                           
Similarly to other sequence transduction models, we use learned embeddings to convert the input tokens and output tokens to vectors of dimension $d_{\text{model}}$.  We also use the usual learned linear transformation and softmax function to convert the decoder output to predicted next-token probabilities.  In our model, we share the same weight matrix between the two embedding layers and the pre-softmax linear transformation, similar to \citep{press2016using}.   In the embedding layers, we multiply those weights by $\sqrt{d_{\text{model}}}$. In addition, we apply dropout to the sums of the embeddings and the positional encodings in both the encoder and decoder stacks.  For the base model, we use a rate of $P_{drop}=0.1$.                                                                                                                                 


```python
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)
```

## Positional Encoding                                                                                                                             
Since our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, we must inject some information about the relative or absolute position of the tokens in the sequence.  To this end, we add "positional encodings" to the input embeddings at the bottoms of the encoder and decoder stacks.  The positional encodings have the same dimension $d_{\text{model}}$ as the embeddings, so that the two can be summed.   There are many choices of positional encodings, learned and fixed \citep{JonasFaceNet2017}.                                                                                                                                                                                                                                                                                                                                                
In this work, we use sine and cosine functions of different frequencies:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
$$                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
    PE_{(pos,2i)} = sin(pos / 10000^{2i/d_{\text{model}}}) \\                                                                                                                                                                                                                                                                                                                                                                                                                                      
    PE_{(pos,2i+1)} = cos(pos / 10000^{2i/d_{\text{model}}})                                                                                                                                                                                                                                                                                                                                                                                                                                       
$$                                                                                                                                                                                                                                                        
where $pos$ is the position and $i$ is the dimension.  That is, each dimension of the positional encoding corresponds to a sinusoid.  The wavelengths form a geometric progression from $2\pi$ to $10000 \cdot 2\pi$.  We chose this function because we hypothesized it would allow the model to easily learn to attend by relative positions, since for any fixed offset $k$, $PE_{pos+k}$ can be represented as a linear function of $PE_{pos}$.                                       
                                                                                                                                                                                                                                                    



```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Compute the positional encodings once.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)
```

We also experimented with using learned positional embeddings \citep{JonasFaceNet2017} instead, and found that the two versions produced nearly identical results (see Table~\ref{tab:variations} row (E)).  We chose the sinusoidal version because it may allow the model to extrapolate to sequence lengths longer than the ones encountered during training.    

# Training

This section describes the training regime for our models.



```python
def make_model(src_vocab, tgt_vocab, padding_idx, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    attn = MultiHeadedAttention(h, d_model, dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, copy.deepcopy(attn), copy.deepcopy(ff), dropout), N),
        Decoder(DecoderLayer(d_model, copy.deepcopy(attn), copy.deepcopy(attn), copy.deepcopy(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), copy.deepcopy(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), copy.deepcopy(position)),
        Generator(d_model, tgt_vocab),
        padding_idx
    )
    
    # This is really important.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model
```


```python
def loss_backprop(model, criterion, out, targets, norm):
    out_grad = []
    total = 0
    assert out.size(1) == targets.size(1)
    for i in range(out.size(1)):
        out_column = Variable(out[:, i].data, requires_grad=True)
        gen = model.generator(out_column)
        loss = criterion(gen, targets[:, i]) / norm
        total += loss.data[0]
        loss.backward()
        out_grad.append(out_column.grad.data.clone())
    out_grad = torch.stack(out_grad, dim=1)
    out.backward(gradient=out_grad)
    return total
```

## Training Data and Batching

We trained on the standard WMT 2014 English-German dataset consisting of about 4.5 million sentence pairs.  Sentences were encoded using byte-pair encoding \citep{DBLP:journals/corr/BritzGLL17}, which has a shared source-target vocabulary of about 37000 tokens. For English-French, we used the significantly larger WMT 2014 English-French dataset consisting of 36M sentences and split tokens into a 32000 word-piece vocabulary \citep{wu2016google}.  Sentence pairs were batched together by approximate sequence length.  Each training batch contained a set of sentence pairs containing approximately 25000 source tokens and 25000 target tokens.     


```python
# Specialized batching. This seems to really matter.
global max_src_in_batch, max_tgt_in_batch

def batch_size_fn(new, count, sofar):
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src) + 2)
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 1)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)
```

## Hardware and Schedule                                                                                                                                                                                                   
We trained our models on one machine with 8 NVIDIA P100 GPUs.  For our base models using the hyperparameters described throughout the paper, each training step took about 0.4 seconds.  We trained the base models for a total of 100,000 steps or 12 hours.  For our big models,(described on the bottom line of table \ref{tab:variations}), step time was 1.0 seconds.  The big models were trained for 300,000 steps (3.5 days).

## Optimizer

We used the Adam optimizer~\citep{kingma2014adam} with $\beta_1=0.9$, $\beta_2=0.98$ and $\epsilon=10^{-9}$.  We varied the learning rate over the course of training, according to the formula:                                                                                            
$$                                                                                                                                                                                                                                                                                         
lrate = d_{\text{model}}^{-0.5} \cdot                                                                                                                                                                                                                                                                                                
  \min({step\_num}^{-0.5},                                                                                                                                                                                                                                                                                                  
    {step\_num} \cdot {warmup\_steps}^{-1.5})                                                                                                                                                                                                                                                                               
$$                                                                                                                                                                                             
This corresponds to increasing the learning rate linearly for the first $warmup\_steps$ training steps, and decreasing it thereafter proportionally to the inverse square root of the step number.  We used $warmup\_steps=4000$.                            


```python
class NoamOpt:
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self):
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(self._step ** (-0.5), self._step * self.warmup**(-1.5)))
        
def get_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
```

## Regularization                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
                                                                                                                                                                                                                                                                                                                      
### Label Smoothing

During training, we employed label smoothing of value $\epsilon_{ls}=0.1$ \citep{DBLP:journals/corr/SzegedyVISW15}.  This hurts perplexity, as the model learns to be more unsure, but improves accuracy and BLEU score.  


```python
class LabelSmoothing(nn.Module):
    def __init__(self, size, padding_idx, label_smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.padding_idx = padding_idx
        self.criterion = nn.KLDivLoss(size_average=False)
        one_hot = torch.zeros(1, size)
        one_hot.fill_(label_smoothing / (size - 2))
        one_hot[0][self.padding_idx] = 0
        self.register_buffer('one_hot', one_hot)
        self.confidence = 1.0 - label_smoothing
        
    def forward(self, x, target):
        tdata = target.view(-1).data
        mask = torch.nonzero(tdata == self.padding_idx).squeeze()
        tmp_ = self.one_hot.repeat(tdata.size(0), 1)
        tmp_.scatter_(1, tdata.unsqueeze(1), self.confidence)
        if mask.dim() > 0:
            tmp_.index_fill_(0, mask, 0)
        gtruth = Variable(tmp_, requires_grad=False)
        return self.criterion(x, gtruth)
```

# Putting it all together.


```python
# Load words from IWSLT

#!pip install torchtext spacy
#!python -m spacy download en
#!python -m spacy download de
def load():
    import spacy
    spacy_de = spacy.load('de')
    spacy_en = spacy.load('en')

    def tokenize_de(text):
        return [tok.text for tok in spacy_de.tokenizer(text)]

    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]

    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
    BLANK_WORD = "<blank>"
    SRC = data.Field(tokenize=tokenize_de, pad_token=BLANK_WORD)
    TGT = data.Field(tokenize=tokenize_en, init_token = BOS_WORD, 
                     eos_token = EOS_WORD, pad_token=BLANK_WORD)

    MAX_LEN = 100
    train, val, test = datasets.IWSLT.splits(exts=('.de', '.en'), fields=(SRC, TGT), 
                                             filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and 
                                             len(vars(x)['trg']) <= MAX_LEN)
    MIN_FREQ = 2
    SRC.build_vocab(train.src, min_freq=MIN_FREQ)
    TGT.build_vocab(train.trg, min_freq=MIN_FREQ)

    BATCH_SIZE = 32
    train_iter, val_iter = data.BucketIterator.splits((train, val), batch_size=BATCH_SIZE, device=0,
                                                      repeat=False, sort_key=lambda x: len(x.src))
```


```python
BOS_WORD = '<s>'
EOS_WORD = '</s>'
BLANK_WORD = "<blank>"
SRC = data.Field()
TGT = data.Field(init_token = BOS_WORD, eos_token = EOS_WORD, pad_token=BLANK_WORD) # only target needs BOS/EOS

MAX_LEN = 100
train = datasets.TranslationDataset(path="/n/home00/srush/Data/baseline-1M_train.tok.shuf", 
                                    exts=('.en', '.fr'),
                                    fields=(SRC, TGT), 
                                    filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and 
                                         len(vars(x)['trg']) <= MAX_LEN)
SRC.build_vocab(train.src, max_size=50000)
TGT.build_vocab(train.trg, max_size=50000)
```


```python
def train_epoch(train_iter, model, criterion, opt):
    for i, batch in enumerate(train_iter):
        src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
        out = model.forward(src, trg[:, :-1])
        #print(model.src_embed[0].lut.weight.grad)
        loss = loss_backprop(model, criterion, out, trg[:, 1:], 
                             (trg[:, 1:] != pad_idx).data.sum())
        model_opt.step()
        model_opt.optimizer.zero_grad()
        if i % 100 == 1:
            print(i, loss, model_opt._rate)
```


```python
def valid_epoch(valid_iter, model, criterion):
    for batch in valid_iter:
        src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
        out = model.forward(src, trg[:, :-1])
        loss = loss_backprop(model, criterion, out, trg[:, 1:], 
                             (trg[:, 1:] == pad_idx).data.sum())
```


```python
pad_idx = TGT.vocab.stoi["<blank>"]
print(pad_idx)
model = make_model(len(SRC.vocab), len(TGT.vocab), pad_idx, N=6)
model_opt = get_opt(model)
model.cuda()
```

    1





    EncoderDecoder(
      (encoder): Encoder(
        (layers): ModuleList(
          (0): EncoderLayer(
            (self_attn): MultiHeadedAttention(
              (linears): ModuleList(
                (0): Linear(in_features=512, out_features=512)
                (1): Linear(in_features=512, out_features=512)
                (2): Linear(in_features=512, out_features=512)
                (3): Linear(in_features=512, out_features=512)
              )
            )
            (feed_forward): PositionwiseFeedForward(
              (w_1): Linear(in_features=512, out_features=2048)
              (w_2): Linear(in_features=2048, out_features=512)
              (dropout): Dropout(p=0.1)
            )
            (sublayer): ModuleList(
              (0): SublayerConnection(
                (norm): LayerNorm(
                )
                (dropout): Dropout(p=0.1)
              )
              (1): SublayerConnection(
                (norm): LayerNorm(
                )
                (dropout): Dropout(p=0.1)
              )
            )
          )
          (1): EncoderLayer(
            (self_attn): MultiHeadedAttention(
              (linears): ModuleList(
                (0): Linear(in_features=512, out_features=512)
                (1): Linear(in_features=512, out_features=512)
                (2): Linear(in_features=512, out_features=512)
                (3): Linear(in_features=512, out_features=512)
              )
            )
            (feed_forward): PositionwiseFeedForward(
              (w_1): Linear(in_features=512, out_features=2048)
              (w_2): Linear(in_features=2048, out_features=512)
              (dropout): Dropout(p=0.1)
            )
            (sublayer): ModuleList(
              (0): SublayerConnection(
                (norm): LayerNorm(
                )
                (dropout): Dropout(p=0.1)
              )
              (1): SublayerConnection(
                (norm): LayerNorm(
                )
                (dropout): Dropout(p=0.1)
              )
            )
          )
          (2): EncoderLayer(
            (self_attn): MultiHeadedAttention(
              (linears): ModuleList(
                (0): Linear(in_features=512, out_features=512)
                (1): Linear(in_features=512, out_features=512)
                (2): Linear(in_features=512, out_features=512)
                (3): Linear(in_features=512, out_features=512)
              )
            )
            (feed_forward): PositionwiseFeedForward(
              (w_1): Linear(in_features=512, out_features=2048)
              (w_2): Linear(in_features=2048, out_features=512)
              (dropout): Dropout(p=0.1)
            )
            (sublayer): ModuleList(
              (0): SublayerConnection(
                (norm): LayerNorm(
                )
                (dropout): Dropout(p=0.1)
              )
              (1): SublayerConnection(
                (norm): LayerNorm(
                )
                (dropout): Dropout(p=0.1)
              )
            )
          )
          (3): EncoderLayer(
            (self_attn): MultiHeadedAttention(
              (linears): ModuleList(
                (0): Linear(in_features=512, out_features=512)
                (1): Linear(in_features=512, out_features=512)
                (2): Linear(in_features=512, out_features=512)
                (3): Linear(in_features=512, out_features=512)
              )
            )
            (feed_forward): PositionwiseFeedForward(
              (w_1): Linear(in_features=512, out_features=2048)
              (w_2): Linear(in_features=2048, out_features=512)
              (dropout): Dropout(p=0.1)
            )
            (sublayer): ModuleList(
              (0): SublayerConnection(
                (norm): LayerNorm(
                )
                (dropout): Dropout(p=0.1)
              )
              (1): SublayerConnection(
                (norm): LayerNorm(
                )
                (dropout): Dropout(p=0.1)
              )
            )
          )
          (4): EncoderLayer(
            (self_attn): MultiHeadedAttention(
              (linears): ModuleList(
                (0): Linear(in_features=512, out_features=512)
                (1): Linear(in_features=512, out_features=512)
                (2): Linear(in_features=512, out_features=512)
                (3): Linear(in_features=512, out_features=512)
              )
            )
            (feed_forward): PositionwiseFeedForward(
              (w_1): Linear(in_features=512, out_features=2048)
              (w_2): Linear(in_features=2048, out_features=512)
              (dropout): Dropout(p=0.1)
            )
            (sublayer): ModuleList(
              (0): SublayerConnection(
                (norm): LayerNorm(
                )
                (dropout): Dropout(p=0.1)
              )
              (1): SublayerConnection(
                (norm): LayerNorm(
                )
                (dropout): Dropout(p=0.1)
              )
            )
          )
          (5): EncoderLayer(
            (self_attn): MultiHeadedAttention(
              (linears): ModuleList(
                (0): Linear(in_features=512, out_features=512)
                (1): Linear(in_features=512, out_features=512)
                (2): Linear(in_features=512, out_features=512)
                (3): Linear(in_features=512, out_features=512)
              )
            )
            (feed_forward): PositionwiseFeedForward(
              (w_1): Linear(in_features=512, out_features=2048)
              (w_2): Linear(in_features=2048, out_features=512)
              (dropout): Dropout(p=0.1)
            )
            (sublayer): ModuleList(
              (0): SublayerConnection(
                (norm): LayerNorm(
                )
                (dropout): Dropout(p=0.1)
              )
              (1): SublayerConnection(
                (norm): LayerNorm(
                )
                (dropout): Dropout(p=0.1)
              )
            )
          )
        )
        (norm): LayerNorm(
        )
      )
      (decoder): Decoder(
        (layers): ModuleList(
          (0): DecoderLayer(
            (self_attn): MultiHeadedAttention(
              (linears): ModuleList(
                (0): Linear(in_features=512, out_features=512)
                (1): Linear(in_features=512, out_features=512)
                (2): Linear(in_features=512, out_features=512)
                (3): Linear(in_features=512, out_features=512)
              )
            )
            (src_attn): MultiHeadedAttention(
              (linears): ModuleList(
                (0): Linear(in_features=512, out_features=512)
                (1): Linear(in_features=512, out_features=512)
                (2): Linear(in_features=512, out_features=512)
                (3): Linear(in_features=512, out_features=512)
              )
            )
            (feed_forward): PositionwiseFeedForward(
              (w_1): Linear(in_features=512, out_features=2048)
              (w_2): Linear(in_features=2048, out_features=512)
              (dropout): Dropout(p=0.1)
            )
            (sublayer): ModuleList(
              (0): SublayerConnection(
                (norm): LayerNorm(
                )
                (dropout): Dropout(p=0.1)
              )
              (1): SublayerConnection(
                (norm): LayerNorm(
                )
                (dropout): Dropout(p=0.1)
              )
              (2): SublayerConnection(
                (norm): LayerNorm(
                )
                (dropout): Dropout(p=0.1)
              )
            )
          )
          (1): DecoderLayer(
            (self_attn): MultiHeadedAttention(
              (linears): ModuleList(
                (0): Linear(in_features=512, out_features=512)
                (1): Linear(in_features=512, out_features=512)
                (2): Linear(in_features=512, out_features=512)
                (3): Linear(in_features=512, out_features=512)
              )
            )
            (src_attn): MultiHeadedAttention(
              (linears): ModuleList(
                (0): Linear(in_features=512, out_features=512)
                (1): Linear(in_features=512, out_features=512)
                (2): Linear(in_features=512, out_features=512)
                (3): Linear(in_features=512, out_features=512)
              )
            )
            (feed_forward): PositionwiseFeedForward(
              (w_1): Linear(in_features=512, out_features=2048)
              (w_2): Linear(in_features=2048, out_features=512)
              (dropout): Dropout(p=0.1)
            )
            (sublayer): ModuleList(
              (0): SublayerConnection(
                (norm): LayerNorm(
                )
                (dropout): Dropout(p=0.1)
              )
              (1): SublayerConnection(
                (norm): LayerNorm(
                )
                (dropout): Dropout(p=0.1)
              )
              (2): SublayerConnection(
                (norm): LayerNorm(
                )
                (dropout): Dropout(p=0.1)
              )
            )
          )
          (2): DecoderLayer(
            (self_attn): MultiHeadedAttention(
              (linears): ModuleList(
                (0): Linear(in_features=512, out_features=512)
                (1): Linear(in_features=512, out_features=512)
                (2): Linear(in_features=512, out_features=512)
                (3): Linear(in_features=512, out_features=512)
              )
            )
            (src_attn): MultiHeadedAttention(
              (linears): ModuleList(
                (0): Linear(in_features=512, out_features=512)
                (1): Linear(in_features=512, out_features=512)
                (2): Linear(in_features=512, out_features=512)
                (3): Linear(in_features=512, out_features=512)
              )
            )
            (feed_forward): PositionwiseFeedForward(
              (w_1): Linear(in_features=512, out_features=2048)
              (w_2): Linear(in_features=2048, out_features=512)
              (dropout): Dropout(p=0.1)
            )
            (sublayer): ModuleList(
              (0): SublayerConnection(
                (norm): LayerNorm(
                )
                (dropout): Dropout(p=0.1)
              )
              (1): SublayerConnection(
                (norm): LayerNorm(
                )
                (dropout): Dropout(p=0.1)
              )
              (2): SublayerConnection(
                (norm): LayerNorm(
                )
                (dropout): Dropout(p=0.1)
              )
            )
          )
          (3): DecoderLayer(
            (self_attn): MultiHeadedAttention(
              (linears): ModuleList(
                (0): Linear(in_features=512, out_features=512)
                (1): Linear(in_features=512, out_features=512)
                (2): Linear(in_features=512, out_features=512)
                (3): Linear(in_features=512, out_features=512)
              )
            )
            (src_attn): MultiHeadedAttention(
              (linears): ModuleList(
                (0): Linear(in_features=512, out_features=512)
                (1): Linear(in_features=512, out_features=512)
                (2): Linear(in_features=512, out_features=512)
                (3): Linear(in_features=512, out_features=512)
              )
            )
            (feed_forward): PositionwiseFeedForward(
              (w_1): Linear(in_features=512, out_features=2048)
              (w_2): Linear(in_features=2048, out_features=512)
              (dropout): Dropout(p=0.1)
            )
            (sublayer): ModuleList(
              (0): SublayerConnection(
                (norm): LayerNorm(
                )
                (dropout): Dropout(p=0.1)
              )
              (1): SublayerConnection(
                (norm): LayerNorm(
                )
                (dropout): Dropout(p=0.1)
              )
              (2): SublayerConnection(
                (norm): LayerNorm(
                )
                (dropout): Dropout(p=0.1)
              )
            )
          )
          (4): DecoderLayer(
            (self_attn): MultiHeadedAttention(
              (linears): ModuleList(
                (0): Linear(in_features=512, out_features=512)
                (1): Linear(in_features=512, out_features=512)
                (2): Linear(in_features=512, out_features=512)
                (3): Linear(in_features=512, out_features=512)
              )
            )
            (src_attn): MultiHeadedAttention(
              (linears): ModuleList(
                (0): Linear(in_features=512, out_features=512)
                (1): Linear(in_features=512, out_features=512)
                (2): Linear(in_features=512, out_features=512)
                (3): Linear(in_features=512, out_features=512)
              )
            )
            (feed_forward): PositionwiseFeedForward(
              (w_1): Linear(in_features=512, out_features=2048)
              (w_2): Linear(in_features=2048, out_features=512)
              (dropout): Dropout(p=0.1)
            )
            (sublayer): ModuleList(
              (0): SublayerConnection(
                (norm): LayerNorm(
                )
                (dropout): Dropout(p=0.1)
              )
              (1): SublayerConnection(
                (norm): LayerNorm(
                )
                (dropout): Dropout(p=0.1)
              )
              (2): SublayerConnection(
                (norm): LayerNorm(
                )
                (dropout): Dropout(p=0.1)
              )
            )
          )
          (5): DecoderLayer(
            (self_attn): MultiHeadedAttention(
              (linears): ModuleList(
                (0): Linear(in_features=512, out_features=512)
                (1): Linear(in_features=512, out_features=512)
                (2): Linear(in_features=512, out_features=512)
                (3): Linear(in_features=512, out_features=512)
              )
            )
            (src_attn): MultiHeadedAttention(
              (linears): ModuleList(
                (0): Linear(in_features=512, out_features=512)
                (1): Linear(in_features=512, out_features=512)
                (2): Linear(in_features=512, out_features=512)
                (3): Linear(in_features=512, out_features=512)
              )
            )
            (feed_forward): PositionwiseFeedForward(
              (w_1): Linear(in_features=512, out_features=2048)
              (w_2): Linear(in_features=2048, out_features=512)
              (dropout): Dropout(p=0.1)
            )
            (sublayer): ModuleList(
              (0): SublayerConnection(
                (norm): LayerNorm(
                )
                (dropout): Dropout(p=0.1)
              )
              (1): SublayerConnection(
                (norm): LayerNorm(
                )
                (dropout): Dropout(p=0.1)
              )
              (2): SublayerConnection(
                (norm): LayerNorm(
                )
                (dropout): Dropout(p=0.1)
              )
            )
          )
        )
        (norm): LayerNorm(
        )
      )
      (src_embed): Sequential(
        (0): Embeddings(
          (lut): Embedding(50002, 512)
        )
        (1): PositionalEncoding(
          (dropout): Dropout(p=0.1)
        )
      )
      (tgt_embed): Sequential(
        (0): Embeddings(
          (lut): Embedding(50004, 512)
        )
        (1): PositionalEncoding(
          (dropout): Dropout(p=0.1)
        )
      )
      (generator): Generator(
        (proj): Linear(in_features=512, out_features=50004)
      )
    )




```python
BATCH_SIZE = 4096
class MyIterator(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b
            self.batches = pool(self.data(), self.random_shuffler)
        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size,
                                          self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))

train_iter = MyIterator(train, batch_size=BATCH_SIZE, device=0,
                        repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                        batch_size_fn=batch_size_fn)
```


```python
print(pad_idx)
print(len(SRC.vocab))
```

    1
    50002



```python
torch.save(model, "/n/rush_lab/trans_ipython.pt")
```

    /n/home00/srush/.conda/envs/py3/lib/python3.6/site-packages/torch/serialization.py:158: UserWarning: Couldn't retrieve source code for container of type EncoderDecoder. It won't be checked for correctness upon loading.
      "type " + obj.__name__ + ". It won't be checked "
    /n/home00/srush/.conda/envs/py3/lib/python3.6/site-packages/torch/serialization.py:158: UserWarning: Couldn't retrieve source code for container of type Encoder. It won't be checked for correctness upon loading.
      "type " + obj.__name__ + ". It won't be checked "
    /n/home00/srush/.conda/envs/py3/lib/python3.6/site-packages/torch/serialization.py:158: UserWarning: Couldn't retrieve source code for container of type EncoderLayer. It won't be checked for correctness upon loading.
      "type " + obj.__name__ + ". It won't be checked "
    /n/home00/srush/.conda/envs/py3/lib/python3.6/site-packages/torch/serialization.py:158: UserWarning: Couldn't retrieve source code for container of type MultiHeadedAttention. It won't be checked for correctness upon loading.
      "type " + obj.__name__ + ". It won't be checked "
    /n/home00/srush/.conda/envs/py3/lib/python3.6/site-packages/torch/serialization.py:158: UserWarning: Couldn't retrieve source code for container of type PositionwiseFeedForward. It won't be checked for correctness upon loading.
      "type " + obj.__name__ + ". It won't be checked "
    /n/home00/srush/.conda/envs/py3/lib/python3.6/site-packages/torch/serialization.py:158: UserWarning: Couldn't retrieve source code for container of type SublayerConnection. It won't be checked for correctness upon loading.
      "type " + obj.__name__ + ". It won't be checked "
    /n/home00/srush/.conda/envs/py3/lib/python3.6/site-packages/torch/serialization.py:158: UserWarning: Couldn't retrieve source code for container of type LayerNorm. It won't be checked for correctness upon loading.
      "type " + obj.__name__ + ". It won't be checked "
    /n/home00/srush/.conda/envs/py3/lib/python3.6/site-packages/torch/serialization.py:158: UserWarning: Couldn't retrieve source code for container of type Decoder. It won't be checked for correctness upon loading.
      "type " + obj.__name__ + ". It won't be checked "
    /n/home00/srush/.conda/envs/py3/lib/python3.6/site-packages/torch/serialization.py:158: UserWarning: Couldn't retrieve source code for container of type DecoderLayer. It won't be checked for correctness upon loading.
      "type " + obj.__name__ + ". It won't be checked "
    /n/home00/srush/.conda/envs/py3/lib/python3.6/site-packages/torch/serialization.py:158: UserWarning: Couldn't retrieve source code for container of type Embeddings. It won't be checked for correctness upon loading.
      "type " + obj.__name__ + ". It won't be checked "
    /n/home00/srush/.conda/envs/py3/lib/python3.6/site-packages/torch/serialization.py:158: UserWarning: Couldn't retrieve source code for container of type PositionalEncoding. It won't be checked for correctness upon loading.
      "type " + obj.__name__ + ". It won't be checked "
    /n/home00/srush/.conda/envs/py3/lib/python3.6/site-packages/torch/serialization.py:158: UserWarning: Couldn't retrieve source code for container of type Generator. It won't be checked for correctness upon loading.
      "type " + obj.__name__ + ". It won't be checked "



```python
#weight = torch.ones(len(TGT.vocab))
#weight[pad_idx] = 0
#criterion = nn.NLLLoss(size_average=False, weight=weight.cuda())
criterion = LabelSmoothing(size=len(TGT.vocab), padding_idx=pad_idx, label_smoothing=0.1)
criterion.cuda()
for epoch in range(15):
    train_epoch(train_iter, model, criterion, model_opt)
```

    1 3.269582842476666 0.0005377044714644026
    101 3.300532897672383 0.0005726430336128369
    201 3.3047672072425485 0.0006075815957612711
    301 2.7151080842595547 0.0006425201579097052
    401 2.6975380268413574 0.0006774587200581396
    501 3.051631387323141 0.0007123972822065737
    601 2.554425454698503 0.000747335844355008
    701 2.6254820519825444 0.0007822744065034422
    801 2.868743653933052 0.0008172129686518764
    901 2.5978208642918617 0.0008521515308003106
    1001 2.5955790174775757 0.0008870900929487448
    1101 2.6764775353949517 0.000922028655097179
    1201 2.464000296778977 0.0009569672172456132
    1301 2.0503073083236814 0.0009919057793940475
    1401 2.295472824771423 0.0010268443415424816
    1501 2.245281406212598 0.0010617829036909158
    1601 2.2577588511630893 0.00109672146583935
    1701 2.2232908592559397 0.0011316600279877844
    1801 2.357596361427568 0.0011665985901362186
    1901 2.121352154412307 0.0012015371522846527
    2001 2.5742998471250758 0.001236475714433087
    2101 2.2518509055953473 0.0012714142765815214
    2201 2.2251326659170445 0.0013063528387299553
    2301 2.078994876006618 0.0013412914008783896
    2401 2.068276036065072 0.001376229963026824
    2501 2.31435151558253 0.0013907788851585368
    2601 1.9106871648691595 0.0013738752565588634
    2701 2.183084836578928 0.0013575733592730722
    2801 2.4668076275847852 0.0013418383196400342
    2901 1.963176985620521 0.0013266380295186675
    3001 2.2140520309330896 0.0013119428705609764
    3101 2.6989458349489723 0.0012977254713568687
    3201 2.1293521663174033 0.0012839604929174666
    3301 2.1402786187827587 0.0012706244386700126
    3401 2.041781216394156 0.0012576954857216498
    3501 2.051893091876991 0.0012451533346344698
    3601 1.5498304846696556 0.001232979075358713
    3701 2.763939742697403 0.001221155067309524
    3801 2.7611468499198963 0.0012096648318570434
    3901 1.7321470333263278 0.0011984929557393293
    4001 2.139603299088776 0.0011876250041103701
    4101 2.1966493157087825 0.0011770474421074978
    4201 2.0962203710805625 0.0011667475639689723
    4301 1.9717675620922819 0.0011567134288575545
    4401 2.097687987901736 0.0011469338026529508
    4501 1.9319786678534001 0.001137398105067946
    4601 1.8846281475271098 0.0011280963615221983
    4701 1.9817245414596982 0.0011190191592759865
    4801 1.7659185670199804 0.0011101576073853326
    4901 2.188665813198895 0.0011015033000912066
    5001 2.1391192222399695 0.0010930482833001135
    5101 1.8125874139368534 0.0010847850238522342
    5201 1.6616800595074892 0.0010767063813072288
    5301 1.6544548005331308 0.0010688055820075176
    5401 1.9542939933016896 0.0010610761952049212
    5501 2.218412609123334 0.0010535121110594244
    5601 1.838119359650591 0.001046107520339004
    5701 1.892627771012485 0.0010388568956672375
    5801 2.2462481096954434 0.0010317549741811346
    5901 1.4471426841337234 0.0010247967414755423
    6001 1.9312338004237972 0.0010179774167228303
    6101 1.7303275546291843 0.001011292438867507
    6201 1.8833909621462226 0.0010047374538051973
    6301 1.8943474531406537 0.0009983083024640838


    /n/home00/srush/.conda/envs/py3/lib/python3.6/site-packages/ipykernel_launcher.py:2: DeprecationWarning: generator 'Iterator.__iter__' raised StopIteration
      


    1 1.5940000533591956 0.0009927515780513657
    101 1.7524283765815198 0.0009865483707369156
    201 1.900138527940726 0.0009804600111146078
    301 1.8419977760640904 0.0009744829985071481
    401 1.9621913449373096 0.0009686139798247046
    501 2.226916428655386 0.0009628497416600543
    601 1.7190162097394932 0.0009571872028951208
    701 1.8589106332874508 0.0009516234077802563
    801 1.8107321247807704 0.000946155519450957
    901 1.6531266793608665 0.0009407808138497059
    1001 1.4840005157748237 0.0009354966740233614
    1101 1.7578403616789728 0.0009303005847689719
    1201 1.3920216620899737 0.0009251901276031373
    1301 1.6626927084289491 0.0009201629760320567
    1401 1.7256765578058548 0.0009152168911012566
    1501 1.6049046433763579 0.0009103497172056578
    1601 1.6955451717367396 0.000905559378142174
    1701 1.6796367820352316 0.0009008438733884249
    1801 1.5794002648835885 0.0008962012745924116
    1901 1.9637197174597532 0.0008916297222591652
    2001 1.4656428614398465 0.0008871274226214399
    2101 1.567156056407839 0.0008826926446824871
    2201 1.542241255287081 0.0008783237174198395
    2301 1.690121710913445 0.0008740190271398465
    2401 1.357302049640566 0.0008697770149734477
    2501 1.9049871656461619 0.0008655961745043597
    2601 2.240402895025909 0.0008614750495214811
    2701 1.7940634173137369 0.0008574122318878972
    2801 1.7314323161263019 0.0008534063595194054
    2901 1.6064868164248765 0.0008494561144659686
    3001 1.7515187719254754 0.0008455602210899614
    3101 1.552100334316492 0.0008417174443354889
    3201 1.6221882179379463 0.0008379265880834463
    3301 1.5139061958470847 0.0008341864935873445
    3401 1.6668659402348567 0.0008304960379852562
    3501 2.1993618682026863 0.0008268541328835436
    3601 1.823760490231507 0.0008232597230083089
    3701 1.8189842144493014 0.0008197117849207771
    3801 1.689056838164106 0.0008162093257930558
    3901 1.5656833801185712 0.0008127513822409492
    4001 1.5621904337021988 0.0008093370192107105
    4101 1.4836799805052578 0.0008059653289168093
    4201 1.47899504378438 0.000802635429827976
    4301 1.6922758186701685 0.0007993464656989501
    4401 1.636858390578709 0.000796097604645519
    4501 1.5558803144613194 0.0007928880382605766
    4601 1.5102424336364493 0.00078971698076907
    4701 1.541241532890126 0.0007865836682198282
    4801 1.5931309935403988 0.000783487357712386
    4901 1.2315586884506047 0.0007804273266570247
    5001 1.527937745093368 0.0007774028720663579
    5101 1.31743333209306 0.0007744133098768835
    5201 1.5960889644484269 0.0007714579742990187
    5301 1.4181096099782735 0.0007685362171942096
    5401 1.4596448407392018 0.0007656474074777987
    5501 1.4594163084111642 0.0007627909305463981
    5601 1.62109798315214 0.0007599661877285873
    5701 1.586864550015889 0.0007571725957578231
    5801 1.5062829439411871 0.0007544095862665088
    5901 1.4292167258172412 0.0007516766053002225
    6001 1.4355267270930199 0.0007489731128511653
    6101 1.4162533966591582 0.0007462985824099354
    6201 1.6518787188415445 0.0007436525005347853
    6301 1.5916137372114463 0.0007410343664375577
    1 1.202994157327339 0.0007387531385993765
    101 1.4649722938484047 0.0007361862332058686
    201 1.1459896704182029 0.0007336459004644837
    301 1.417104929103516 0.0007311316850490442
    401 1.373963651509257 0.0007286431424819469
    501 1.6432027550181374 0.0007261798388040814
    601 1.4122836171882227 0.0007237413502569408
    701 1.6119428309611976 0.0007213272629763972
    801 1.5545603609643877 0.0007189371726976359
    901 1.5427279596333392 0.0007165706844707772
    1001 1.5437391183004365 0.0007142274123867243
    1101 1.9743895339342998 0.0007119069793128112
    1201 1.730805973522365 0.0007096090166378355
    1301 1.5635135210759472 0.0007073331640260875
    1401 1.206731209764257 0.000705079069180001
    1501 1.4495476994197816 0.0007028463876110714
    1601 1.2935033895773813 0.0007006347824187037
    1701 1.1734203454107046 0.000698443924076667
    1801 1.202259551268071 0.0006962734902268488
    1901 1.7874216835407424 0.0006941231654800159
    2001 1.5438914835685864 0.0006919926412233024
    2101 1.5168145569041371 0.000689881615434157
    2201 1.5306344364071265 0.0006877897925004977
    2301 1.5227781175635755 0.0006857168830468271
    2401 1.3308223116910085 0.0006836626037660786
    2501 1.4871021673316136 0.0006816266772569715
    2601 1.3415705130901188 0.0006796088318666611
    2701 1.2746119699440897 0.0006776088015384847
    2801 1.3439618053671438 0.0006756263256646049
    2901 1.3065503026737133 0.0006736611489433701
    3001 1.4918707825127058 0.0006717130212412112
    3101 1.4003087060991675 0.0006697816974589058
    3201 1.3473156996478792 0.0006678669374020495
    3301 1.3869949235959211 0.0006659685056555759
    3401 1.5086751837225165 0.000664086171462178
    3501 1.4735991460911464 0.00066221970860449
    3601 1.3997832712557283 0.0006603688952908887
    3701 1.5196008981074556 0.0006585335140447885
    3801 1.2834229312138632 0.0006567133515973014
    3901 1.3874705795169575 0.0006549081987831418
    4001 1.6422591609880328 0.0006531178504396635
    4101 1.305389653716702 0.0006513421053089143
    4201 1.5159487561322749 0.0006495807659426053
    4301 1.3981374967552256 0.0006478336386098913
    4401 1.7390631912276149 0.0006461005332078655
    4501 1.3604947600979358 0.0006443812631746732
    4601 1.7799529591429746 0.000642675645405156
    4701 1.3463407127128448 0.0006409835001689394
    4801 1.4632963918847963 0.0006393046510308797
    4901 1.1903231081087142 0.0006376389247737917
    5001 1.3287691511941375 0.0006359861513233783
    5101 1.3445309301023372 0.0006343461636752915
    5201 1.5431754024625661 0.0006327187978242499
    5301 1.3343850841192761 0.0006311038926951474
    5401 1.1768817943520844 0.0006295012900760858
    5501 1.6530805606771537 0.0006279108345532683
    5601 1.2646167293241888 0.000626332373447694
    5701 1.3651119051501155 0.000624765756753594
    5801 1.831987822048177 0.0006232108370785525
    5901 1.3451470380132378 0.0006216674695852594
    6001 1.5295006221767835 0.0006201355119348414
    6101 1.2796215488779126 0.0006186148242317232
    6201 1.3307579715619795 0.0006171052689699666
    6301 1.5296110774725094 0.0006156067109810445
    1 1.355640010209754 0.0006142969713181733
    101 1.3438594869803637 0.0006128187302418007
    201 1.3398014856502414 0.0006113511097561582
    301 1.2453488917089999 0.0006098939832926246
    401 1.74672898178801 0.0006084472263842588
    501 1.348103358541266 0.0006070107166211413
    601 1.2492765338683967 0.0006055843336068713
    701 1.568915182055207 0.0006041679589161831
    801 1.3617599749704823 0.0006027614760536461
    901 1.3296397840604186 0.0006013647704134199
    1001 1.506301498040557 0.0005999777292400283
    1101 1.1846984136500396 0.000598600241590126
    1201 1.1235853107646108 0.0005972321982952243
    1301 1.3506322290195385 0.0005958734919253515
    1401 1.5431637589354068 0.0005945240167536175
    1501 1.4227895765798166 0.0005931836687216574
    1601 1.2444980588334147 0.0005918523454059284
    1701 1.37204463215312 0.000590529945984835
    1801 1.3662666375166737 0.0005892163712066582
    1901 1.758998476434499 0.0005879115233582672
    2001 1.3996043455335894 0.0005866153062345879
    2101 1.409632071852684 0.0005853276251088103
    2201 1.3139934270293452 0.0005840483867033116
    2301 1.2863777373568155 0.0005827774991612753
    2401 1.1966209802776575 0.0005815148720189864
    2501 1.3174833165830933 0.0005802604161787846
    2601 1.406668136944063 0.0005790140438826557
    2701 1.31760111481708 0.0005777756686864456
    2801 1.22686495014932 0.0005765452054346768
    2901 1.4871160766715548 0.0005753225702359537
    3001 1.3321835576352896 0.0005741076804389384
    3101 1.349290698301047 0.0005729004546088824
    3201 1.0498975263908505 0.0005717008125046992
    3301 1.4295434548403136 0.0005705086750565621
    3401 1.3862976277887356 0.0005693239643440145
    3501 1.3612052928074263 0.0005681466035745775
    3601 1.3539716337691061 0.0005669765170628427
    3701 1.3053378225304186 0.0005658136302100359
    3801 1.2067344364186283 0.0005646578694840415
    3901 1.417662046442274 0.0005635091623998715
    4001 1.2578378450434684 0.0005623674375005725
    4101 1.2363171717152 0.0005612326243385544
    4201 1.3426340871083084 0.0005601046534573332
    4301 1.3097076122212457 0.000558983456373675
    4401 1.0131576862186193 0.0005578689655601316
    4501 1.4332989812392043 0.000556761114427959
    4601 1.4043821960221976 0.0005556598373104054
    4701 1.373746110650245 0.0005545650694463629
    4801 1.2657524709356949 0.0005534767469643717
    4901 1.1224889098666608 0.0005523948068669684
    5001 1.2615516305086203 0.000551319187015369
    5101 1.409785834257491 0.0005502498261144795
    5201 1.3791224808810512 0.0005491866636982242
    5301 1.2408291140163783 0.0005481296401151859
    5401 1.3008261130889878 0.0005470786965145471
    5501 1.1700160388209042 0.0005460337748323287
    5601 1.2999350049067289 0.000544994817777915
    5701 1.3322585223941132 0.0005439617688208604
    5801 1.254337038320955 0.0005429345721779703
    5901 1.773689029644629 0.000541913172800649
    6001 1.3898115772462916 0.0005408975163625087
    6101 1.4735579792177305 0.0005398875492472326
    6201 1.05738219874911 0.000538883218536687
    6301 1.0802461032290012 0.0005378844719992749
    1 1.286231731530279 0.0005370101533168812
    101 1.2250633136718534 0.0005360217659787991
    201 1.239320948603563 0.0005350388161199592
    301 1.4140636462761904 0.0005340612540665886
    401 1.442663955502212 0.0005330890307779102
    501 1.4505203103472013 0.0005321220978358095
    601 1.2115196966333315 0.0005311604074347066
    701 1.2035035027656704 0.0005302039123716286
    801 1.3747974793659523 0.0005292525660364788
    901 1.36490419106849 0.0005283063224024965
    1001 1.1864821948111057 0.0005273651360169036
    1101 1.1623371304303873 0.000526428961991735
    1201 1.1043747729854658 0.0005254977559948457
    1301 1.6982813560443901 0.0005245714742410941
    1401 1.2719842366641387 0.0005236500734836944
    1501 1.2951120301149786 0.0005227335110057353
    1601 1.580276207998395 0.0005218217446118628
    1701 1.218743062199792 0.0005209147326201215
    1801 1.1479590674862266 0.0005200124338539494
    1901 1.2872504810075043 0.0005191148076343284
    2001 1.8993003838438653 0.0005182218137720798
    2101 1.2762204335303977 0.0005173334125603075
    2201 1.6183682525045242 0.0005164495647669814
    2301 1.2522982619411778 0.0005155702316276618
    2401 1.2925108795752749 0.0005146953748383575
    2501 1.340747339767404 0.0005138249565485178
    2601 1.340512964350637 0.0005129589393541545
    2701 1.1672844442073256 0.0005120972862910908
    2801 1.257948145037517 0.0005112399608283344
    2901 1.510728154462413 0.0005103869268615725
    3001 1.4130934766726568 0.0005095381487067851
    3101 1.2367545471934136 0.0005086935910939762
    3201 1.3846962348325178 0.0005078532191610173
    3301 1.2582954101526411 0.0005070169984476032
    3401 1.1545094328466803 0.0005061848948893172
    3501 1.295005505089648 0.0005053568748118022
    3601 1.3319955187034793 0.0005045329049250373
    3701 1.3548947679810226 0.000503712952317716
    3801 1.4635376840888057 0.0005028969844517252
    3901 1.6542128916307774 0.0005020849691567213
    4001 1.3512894048908493 0.0005012768746248036
    4101 1.397591198408918 0.0005004726694052806
    4201 1.3055676214280538 0.0004996723223995292
    4301 1.3375271083787084 0.000498875802855943
    4401 1.2366086341207847 0.0004980830803649704
    4501 1.2439679206581786 0.0004972941248542376
    4601 1.352382222772576 0.0004965089065837576
    4701 1.7570512742054234 0.0004957273961412208
    4801 1.232903058291413 0.0004949495644373684
    4901 1.015858386293985 0.0004941753827014446
    5001 1.381107110035373 0.000493404822476726
    5101 0.9564947709441185 0.0004926378556161293
    5201 1.228621664486127 0.0004918744542778926
    5301 1.182083563413471 0.0004911145909213302
    5401 1.2583643229590962 0.0004903582383026592
    5501 1.404046923678834 0.0004896053694708976
    5601 1.2389367091745953 0.0004888559577638302
    5701 1.119320425321348 0.00048810997680404295
    5801 1.586507015679672 0.0004873674004950231
    5901 1.112720330056618 0.0004866282030173253
    6001 1.3577893248293549 0.0004858923588248005
    6101 1.217524498468265 0.0004851598426408882
    6201 1.3229771983387764 0.0004844306294549693
    6301 1.5693217546272535 0.0004837046945187796
    1 1.1786362157727126 0.00048306134975017534
    101 1.28241519164294 0.00048234154403106603
    201 1.1411214591062162 0.00048162494648183897
    301 1.2352831599419005 0.0004809115333417623
    401 1.1032181181944907 0.00048020128109574806
    501 1.18390864826506 0.00047949416647109663
    601 1.2226583541632863 0.00047879016643429347
    701 1.0373018080717884 0.0004780892581878584
    801 1.2819566036341712 0.00047739141916724456
    901 1.1648676298791543 0.0004766966270377871
    1001 1.1654199322802015 0.00047600485969170105
    1101 1.2386636545270449 0.00047531609524512704
    1201 1.2253044219105504 0.0004746303120352227
    1301 1.375744077755371 0.0004739474886173019
    1401 1.1551300736318808 0.0004732676037620178
    1501 1.5255512128351256 0.00047259063645259034
    1601 1.255034319277911 0.00047191656588207824
    1701 1.1623500876303297 0.0004712453714506923
    1801 1.2958592986833537 0.00047057703276315175
    1901 1.1341320046922192 0.0004699115296260807
    2001 1.1937441515619867 0.0004692488420454462
    2101 1.7062073841661913 0.00046858895022403485
    2201 1.2566360468044877 0.00046793183455896863
    2301 1.2216275975806639 0.0004672774756392595
    2401 1.2636524712725077 0.0004666258542434008
    2501 1.2113699619076215 0.00046597695133699556
    2601 1.1559934263350442 0.00046533074807042176
    2701 1.256740387296304 0.0004646872257765319
    2801 1.3039579528664262 0.0004640463659683885
    2901 1.2651300196012016 0.0004634081503370334
    3001 1.2652980692801066 0.000462772560749291
    3101 1.1218284339411184 0.00046213957924560355
    3201 1.2543016897689085 0.0004615091880379007
    3301 1.2131407480192138 0.0004608813695074994
    3401 1.2994702684518415 0.0004602561062030357
    3501 1.2115506358095445 0.00045963338083842724
    3601 1.1760960748360958 0.00045901317629086643
    3701 1.0682971130590886 0.00045839547559884254
    3801 1.0764332090620883 0.00045778026196019347
    3901 1.1835216325707734 0.0004571675187301866
    4001 1.3529939632862806 0.00045655722941962654
    4101 1.3684578015236184 0.0004559493776929923
    4201 1.2233722301607486 0.0004553439473666001
    4301 1.2596116681525018 0.0004547409224067939
    4401 1.2757911044172943 0.00045414028692816196
    4501 1.2199301174841821 0.0004535420251917793
    4601 1.3471774608151463 0.0004529461216034753
    4701 1.475795219448628 0.0004523525607121267
    4801 1.1835241899825633 0.0004517613272079745
    4901 1.1791616377497576 0.0004511724059209659
    5001 1.3126113665202865 0.0004505857818191191
    5101 1.2516068609402282 0.0004500014400069121
    5201 1.178165558274486 0.0004494193657236937
    5301 1.6013869942435122 0.0004488395443421177
    5401 1.2677101592962572 0.00044826196136659916
    5501 1.1976667390699731 0.0004476866024317922
    5601 1.1990807302790927 0.00044711345330108884
    5701 1.1415361673789448 0.0004465424998651406
    5801 1.2389779405202717 0.0004459737281403985
    5901 1.1746156329172663 0.0004454071242676752
    6001 1.1718775559565984 0.0004448426745107265
    6101 1.1669323876558337 0.00044428036525485275
    6201 1.22836275130976 0.0004437201830055194
    6301 1.1068585112225264 0.000443162114386997
    1 1.1908240653865505 0.00044267275186678196
    101 1.156728027795907 0.0004421186210736662
    201 1.151486962888157 0.0004415665660409348
    301 1.1075408830074593 0.0004410165738412884
    401 1.1251853418070823 0.00044046863165985925
    501 1.224421168473782 0.0004399227267929559
    601 1.1097798637929372 0.00043937884664682695
    701 0.992531725903973 0.0004388369787364407
    801 1.2762772621936165 0.0004382971106842813
    901 1.154728337773122 0.00043775923021916087
    1001 0.9699444866273552 0.00043722332517504866
    1101 1.1039727496681735 0.0004366893834899152
    1201 1.2997219555545598 0.0004361573932045913
    1301 1.5713044246076606 0.00043562734246164385
    1401 1.1782071397465188 0.00043509921950426545
    1501 1.256332863289117 0.0004345730126751789
    1601 1.162631830346072 0.00043404871041555687
    1701 1.1123517343075946 0.00043352630126395546
    1801 1.0946980192093179 0.0004330057738552615
    1901 1.120711475959979 0.0004324871169196544
    2001 1.1385652619646862 0.0004319703192815812
    2101 1.0391206528292969 0.0004314553698587452
    2201 1.1468603002722375 0.00043094225766110786
    2301 1.1944863148819422 0.0004304309717899036
    2401 1.1445480604888871 0.00042992150143666746
    2501 1.160092411795631 0.0004294138358822756
    2601 0.905779943568632 0.00042890796449599795
    2701 1.2337692737637553 0.0004284038767345632
    2801 1.2654334787439439 0.00042790156214123586
    2901 1.2613030684588011 0.0004274010103449054
    3001 1.1566388571663992 0.0004269022110591865
    3101 1.1506170178181492 0.0004264051540815317
    3201 1.1042177192866802 0.00042590982929235444
    3301 1.268968387885252 0.00042541622665416415
    3401 1.1708880871301517 0.0004249243362107117
    3501 1.1094016103306785 0.00042443414808614573
    3601 1.3188527839665767 0.0004239456524841804
    3701 1.2144307589042 0.0004234588396872726
    3801 1.1827894128946355 0.0004229737000558104
    3901 0.9924444004427642 0.00042249022402731095
    4001 1.1228576390712988 0.0004220084021156294
    4101 1.1924936635477934 0.0004215282249101765
    4201 1.1275967326655518 0.0004210496830751471
    4301 1.0625419117277488 0.0004205727673487576
    4401 1.1389823842037003 0.00042009746854249313
    4501 1.339291847194545 0.00041962377754036395
    4601 1.0302886090357788 0.0004191516852981713
    4701 1.7122778899138211 0.00041868118284278167
    4801 1.2910672437865287 0.0004182122612714111
    4901 1.0494152382598259 0.00041774491175091685
    5001 1.1782474033534527 0.0004172791255170995
    5101 1.040663594380021 0.0004168148938740118
    5201 0.9901199785526842 0.00041635220819327733
    5301 1.5490817801910453 0.00041589105991341656
    5401 1.1346296942792833 0.00041543144053918197
    5501 1.223581779631786 0.00041497334164089994
    5601 0.958975835936144 0.0004145167548538224
    5701 1.238148811913561 0.0004140616718774844
    5801 1.1881117207813077 0.000413608084475071
    5901 1.1295715225860476 0.0004131559844727907
    6001 1.0610488005331717 0.000412705363759257
    6101 1.2371235750615597 0.00041225621428487707
    6201 1.1479767516138963 0.00041180852806124783
    6301 1.2059640220250003 0.0004113622971605593
    1 1.1765662879188312 0.0004109663692823915
    101 1.219779463717714 0.0004105228675028437
    201 0.972488499362953 0.00041008079847008953
    301 1.1617506150214467 0.0004096401544864815
    401 1.0883347367926035 0.0004092009279121472
    501 1.1085488446406089 0.00040876311116443343
    601 1.1023443718672752 0.0004083266967173559
    701 1.018611608800711 0.00040789167710105623
    801 1.1658196483003849 0.00040745804490126497
    901 1.2917855954219704 0.0004070257927587706
    1001 1.186474629881559 0.00040659491336889525
    1101 1.127356821874855 0.00040616539948097586
    1201 1.1975307842949405 0.00040573724389785204
    1301 1.1174790017685154 0.0004053104394753595
    1401 1.0863252188792103 0.0004048849791218294
    1501 1.0622602235816885 0.000404460855797593
    1601 1.1195327076129615 0.00040403806251449327
    1701 1.2447982146404684 0.00040361659233540054
    1801 1.2607627244724426 0.0004031964383737348
    1901 1.278331945562968 0.00040277759379299307
    2001 1.025594950420782 0.0004023600518062819
    2101 1.115274733179831 0.0004019438056758561
    2201 1.1167924739420414 0.0004015288487126612
    2301 1.0995151306560729 0.000401115174275883
    2401 1.1547567544039339 0.00040070277577250023
    2501 1.1590442548913416 0.00040029164665684384
    2601 1.1047132272506133 0.00039988178043016053
    2701 1.0620037270709872 0.00039947317064018093
    2801 1.0939110746548977 0.00039906581088069363
    2901 0.9693534299731255 0.0003986596947911227
    3001 1.2754340882529505 0.0003982548160561108
    3101 2.0011723663365046 0.0003978511684051071
    3201 1.1672507325711194 0.0003974487456119586
    3301 0.8547125565819442 0.00039704754149450736
    3401 1.0948779656609986 0.00039664754991419163
    3501 1.1662541554399013 0.0003962487647756509
    3601 1.242357063729287 0.00039585118002633614
    3701 1.142975198366912 0.000395454789656124
    3801 1.2055247909738682 0.0003950595876969351
    3901 1.2129587295930833 0.0003946655682223565
    4001 1.239052205113694 0.0003942727253472687
    4101 1.1285374546132516 0.0003938810532274764
    4201 1.3123125492420513 0.00039349054605934306
    4301 1.2088057449145708 0.00039310119807943006
    4401 1.0897887839237228 0.00039271300356413926
    4501 1.2131327866227366 0.00039232595682935973
    4601 0.9877158605959266 0.000391940052230118
    4701 1.2145014504967548 0.0003915552841602323
    4801 1.1513765653944574 0.0003911716470519708
    4901 1.1751896993955597 0.0003907891353757127
    5001 1.1771067604749987 0.0003904077436396139
    5101 1.3224336538114585 0.0003900274663892758
    5201 1.355893952131737 0.0003896482982074174
    5301 1.1996791153214872 0.0003892702337135512
    5401 1.206806231304654 0.0003888932675636631
    5501 1.280991047504358 0.0003885173944498942
    5601 0.9168080711970106 0.0003881426091002278
    5701 1.1924245768459514 0.0003877689062781782
    5801 1.1940782586170826 0.0003873962807824836
    5901 1.2784329915011767 0.0003870247274468023
    6001 1.1448089724872261 0.00038665424113941134
    6101 1.1181278676813236 0.0003862848167629092
    6201 1.2686003018752672 0.00038591644925392126
    6301 1.0795898175565526 0.000385549133582808
    1 1.199639980099164 0.00038523407955521927
    101 1.0967486363369972 0.00038486870703897236
    201 1.2039306252845563 0.00038450437215947677
    301 1.106695241353009 0.0003841410700146326
    401 1.0133273621067929 0.00038377879573470126
    501 1.1209047999582253 0.0003834175444820315
    601 1.073817516444251 0.00038305731145078797
    701 1.1002086726948619 0.00038269809186668256
    801 1.044247637852095 0.00038233988098670897
    901 1.1538543488713913 0.00038198267409887953
    1001 1.1638364950194955 0.00038162646652196454
    1101 1.1222136826545466 0.00038127125360523515
    1201 1.0936923912668135 0.0003809170307282081
    1301 1.0790816302178428 0.0003805637933003932
    1401 1.0973517978854943 0.0003802115367610436
    1501 1.3543376340385294 0.00037986025657890806
    1601 1.0638599513913505 0.0003795099482519871
    1701 1.095864930888638 0.0003791606073072896
    1801 1.3785420355270617 0.00037881222930059356
    1901 1.0656100183841772 0.0003784648098162084
    2001 1.083574770949781 0.0003781183444667399
    2101 1.7192173111798184 0.0003777728288928577
    2201 1.090653446619399 0.0003774282587630644
    2301 1.1122783308528597 0.00037708462977346826
    2401 0.95696423901245 0.0003767419376475568
    2501 1.2160079335735645 0.0003764001781359734
    2601 1.2086039673304185 0.00037605934701629616
    2701 1.0832101813866757 0.00037571944009281874
    2801 1.013074157119263 0.00037538045319633314
    2901 1.0823434699559584 0.00037504238218391556
    3001 1.1612105248786975 0.0003747052229387128
    3101 0.9126896761590615 0.0003743689713697328
    3201 1.3341417479550728 0.00037403362341163505
    3301 0.9600269035436213 0.0003736991750245252
    3401 1.1916928359714802 0.0003733656221937497
    3501 1.4976106537505984 0.0003730329609296942
    3601 1.1840611910447478 0.0003727011872675824
    3701 1.3150727476167958 0.0003723702972672783
    3801 1.221188339870423 0.00037204028701308904
    3901 1.2026138188084587 0.0003717111526135708
    4001 1.3793403076779214 0.00037138289020133557
    4101 1.0772470782976598 0.0003710554959328607
    4201 1.0168679640773917 0.0003707289659882998
    4301 1.1685322925950459 0.00037040329657129513
    4401 1.1224966624286026 0.00037007848390879306
    4501 1.0253048577578738 0.00036975452425085955
    4601 1.2101853350850433 0.00036943141387049916
    4701 1.050108958443161 0.0003691091490634741
    4801 1.2717001468117815 0.00036878772614812674
    4901 1.1231291381409392 0.00036846714146520227
    5001 1.1141781120968517 0.00036814739137767423
    5101 0.9994667179416865 0.00036782847227057074
    5201 1.3557415267750912 0.0003675103805508032
    5301 1.504937146051816 0.0003671931126469962
    5401 1.0444834220979828 0.00036687666500931896
    5501 1.0159150707913795 0.0003665610341093186
    5601 1.4691102042561397 0.00036624621643975515
    5701 1.2679062836105004 0.0003659322085144373
    5801 1.1070539963402553 0.0003656190068680607
    5901 1.2043958652066067 0.0003653066080560474
    6001 1.1217296464601532 0.00036499500865438625
    6101 1.2132740695233224 0.00036468420525947586
    6201 1.452793362134571 0.00036437419448796804
    6301 1.0731251265387982 0.00036406497297661317
    1 1.1998733360724145 0.00036379350826718935
    101 1.1498671763110906 0.0003634857615296514
    201 1.1022596344992053 0.00036317879447701637
    301 1.0011312331771478 0.00036287260382257964
    401 1.0373523531015962 0.0003625671862990008
    501 1.0644397677097004 0.000362262538658157
    601 1.1183270415349398 0.000361958657670998
    701 0.9439565273642074 0.00036165554012740277
    801 1.0483181119780056 0.0003613531828360362
    901 1.10791165966657 0.00036105158262420917
    1001 0.9895736404578201 0.00036075073633773743
    1101 1.0942751504917396 0.00036045064084080426
    1201 1.0274720180314034 0.0003601512930158222
    1301 1.049829078532639 0.0003598526897632977
    1401 1.0599873741302872 0.00035955482800169595
    1501 1.1110646773013286 0.00035925770466730756
    1601 1.1195740707335062 0.0003589613167141159
    1701 1.1175601889844984 0.0003586656611136664
    1801 1.27650167158572 0.00035837073485493607
    1901 1.1153064398095012 0.000358076534944205
    2001 1.4756392514391337 0.000357783058404929
    2101 0.8967400579713285 0.0003574903022776121
    2201 1.0554919667192735 0.0003571982636196827
    2301 1.1484477042686194 0.000356906939505368
    2401 1.0967925000586547 0.00035661632702557175
    2501 1.275310180048109 0.0003563264232877516
    2601 1.084061863599345 0.00035603722541579873
    2701 1.076280384673737 0.00035574873054991784
    2801 1.200010517553892 0.0003554609358465082
    2901 1.059172057226533 0.000355173838478046
    3001 1.0261963941156864 0.000354887435632968
    3101 0.9737269952893257 0.0003546017245155551
    3201 1.109491402952699 0.0003543167023458187
    3301 1.0379895093501545 0.0003540323663593864
    3401 1.0210047286818735 0.00035374871380738974
    3501 1.3062616163679195 0.0003534657419563522
    3601 1.1297821700572968 0.00035318344808807914
    3701 1.095454223890556 0.0003529018294995477
    3801 1.0263875699602067 0.00035262088350279793
    3901 1.0200049103004858 0.00035234060742482575
    4001 1.139240143907955 0.00035206099860747537
    4101 1.185085133272878 0.00035178205440733397
    4201 1.0887831579057092 0.0003515037721956263
    4301 1.7533796445081862 0.000351226149358111
    4401 1.1149956742010545 0.00035094918329497705
    4501 1.1544227619015146 0.0003506728714207421
    4601 1.1121961249154992 0.00035039721116415036
    4701 1.0929120010696352 0.0003501221999680728
    4801 1.118996370700188 0.000349847835289407
    4901 0.9860638748505153 0.00034957411459897886
    5001 1.004449057742022 0.0003493010353814441
    5101 1.2927988782757893 0.00034902859513519165
    5201 0.98420900356723 0.0003487567913722472
    5301 1.1210692654858576 0.000348485621618178
    5401 1.163566045666812 0.00034821508341199766
    5501 1.17701395630138 0.0003479451743060731
    5601 1.0291424575298151 0.00034767589186603104
    5701 1.2543358486509533 0.0003474072336706659
    5801 1.2299754508348997 0.00034713919731184855
    5901 1.1936145080917413 0.0003468717803944353
    6001 0.9138605990447104 0.00034660498053617827
    6101 1.1037570714397589 0.00034633879536763624
    6201 1.04157008238235 0.00034607322253208626
    6301 1.1919174637878314 0.0003458082596854357
    1 0.9797578346915543 0.00034557559511139293
    101 0.9891712895187084 0.00034531177274179953
    201 1.1815550197497942 0.00034504855367990236
    301 1.1358435613219626 0.0003447859356297997
    401 1.0717519058380276 0.000344523916307803
    501 1.172375235328218 0.00034426249344235384
    601 1.0603420100815129 0.00034400166477394084
    701 1.0992282917286502 0.000343741428055018
    801 1.04559408465866 0.0003434817810499231
    901 1.0248865495998416 0.0003432227215347973
    1001 1.0598302248690743 0.0003429642472975047
    1101 1.0938185814011376 0.0003427063561375535
    1201 1.291374852447916 0.000342449045866017
    1301 1.0285806620959193 0.0003421923143054557
    1401 1.17992360109929 0.0003419361592898398
    1501 0.9615428688703105 0.0003416805786644727
    1601 1.2486475716141285 0.0003414255702859144
    1701 0.9794061238644645 0.0003411711320219065
    1801 1.1059001302346587 0.00034091726175129706
    1901 0.9131126408465207 0.00034066395736396637
    2001 0.9930663524428383 0.0003404112167607534
    2101 1.0812218267819844 0.0003401590378533823
    2201 1.0715046060213353 0.0003399074185643907
    2301 1.1185214965953492 0.00033965635682705713
    2401 1.05721441534115 0.00033940585058533
    2501 1.0936700437378022 0.00033915589779375693
    2601 1.07034515045234 0.00033890649641741454
    2701 1.0813248989579733 0.00033865764443183875
    2801 1.0510470166627783 0.0003384093398229561
    2901 0.9623011860530823 0.0003381615805870148
    3001 1.1494725269731134 0.00033791436473051725
    3101 1.2335324875239166 0.0003376676902701525
    3201 1.0398004396120086 0.00033742155523272933
    3301 1.0589452146668918 0.0003371759576551101
    3401 1.084106142167002 0.00033693089558414497
    3501 1.0406659920408856 0.0003366863670766065
    3601 0.9325393754988909 0.00033644237019912526
    3701 1.0507557194505353 0.0003361989030281253
    3801 0.945562198292464 0.0003359559636497606
    3901 1.0489263081690297 0.0003357135501598519
    4001 1.0528855019947514 0.00033547166066382383
    4101 1.1952165458351374 0.0003352302932766432
    4201 1.0958911241032183 0.00033498944612275674
    4301 1.077680416405201 0.0003347491173360301
    4401 1.2972540742484853 0.0003345093050596873
    4501 1.039087069220841 0.0003342700074462501
    4601 0.9597453814931214 0.00033403122265747876
    4701 1.1728105103829876 0.00033379294886431207
    4801 1.1258336059836438 0.0003335551842468092
    4901 1.0011352995352354 0.0003333179269940906
    5001 0.9672558718448272 0.00033308117530428074
    5101 1.0522874586749822 0.0003328449273844502
    5201 1.0063609674107283 0.0003326091814505589
    5301 1.0480196221014921 0.00033237393572739917
    5401 1.0445173801199417 0.00033213918844854004
    5501 1.417743748796056 0.00033190493785627127
    5601 1.0902981872641249 0.000331671182201548
    5701 1.0301871165866032 0.00033143791974393625
    5801 1.4090074983541854 0.00033120514875155805
    5901 1.1904211936052889 0.0003309728675010378
    6001 1.1052953382313717 0.0003307410742774485
    6101 1.133972127106972 0.00033050976737425853
    6201 1.4820798086614104 0.00033027894509327907
    6301 1.1476092239608988 0.00033004860574461153
    1 1.0870586273958907 0.00032984630526377065
    101 1.00110832543578 0.00032961686928176145
    201 0.9876259700540686 0.0003293879114110055
    301 1.2413200094233616 0.0003291594299932822
    401 0.9424758276436478 0.00032893142337841173
    501 1.0926933737646323 0.00032870388992420444
    601 1.0706572374765528 0.0003284768279964114
    701 1.0879125816572923 0.00032825023596867546
    801 1.262531905740616 0.0003280241122224816
    901 1.050877535046311 0.0003277984551471088
    1001 1.135452825037646 0.0003275732631395822
    1101 1.0099836179433623 0.0003273485346046242
    1201 1.1641883124248125 0.0003271242679546084
    1301 1.0249766572378576 0.00032690046160951133
    1401 1.216345368164184 0.0003266771139968662
    1501 1.4009095890432945 0.00032645422355171653
    1601 0.9214687866624445 0.00032623178871657
    1701 1.050587208737852 0.0003260098079413526
    1801 1.0106142781150993 0.0003257882796833635
    1901 0.9388233295176178 0.00032556720240723
    2001 1.1458081254386343 0.0003253465745848626
    2101 1.0818304931126477 0.00032512639469541087
    2201 0.8635702040046453 0.0003249066612252194
    2301 1.0180434776411857 0.00032468737266778394
    2401 1.0467977939988486 0.0003244685275237081
    2501 1.083000476603047 0.0003242501243006605
    2601 1.1069668279960752 0.00032403216151333166
    2701 1.086160118225962 0.00032381463768339173
    2801 1.0924749624973629 0.0003235975513394485
    2901 1.009744831302669 0.00032338090101700554
    3001 0.9085385013604537 0.0003231646852584205
    3101 1.1706041378201917 0.00032294890261286426
    3201 1.032271361502353 0.00032273355163627964
    3301 1.2584509218577296 0.00032251863089134133
    3401 1.2436874122358859 0.000322304138947415
    3501 1.0451832013077365 0.0003220900743805179
    3601 1.0900762653473066 0.00032187643577327854
    3701 1.1192542002827395 0.00032166322171489793
    3801 1.00253647408681 0.0003214504308011099
    3901 1.2160937447333708 0.0003212380616341424
    4001 1.0416435159859248 0.0003210261128226793
    4101 1.3598752447869629 0.00032081458298182156
    4201 1.0555532689650136 0.0003206034707330495
    4301 1.1295962483854964 0.00032039277470418526
    4401 0.9410244208120275 0.0003201824935293548
    4501 0.8939700378105044 0.00031997262584895135
    4601 0.908640876179561 0.000319763170309598
    4701 1.128680162204546 0.0003195541255641112
    4801 1.0497526655672118 0.00031934549027146444
    4901 1.0289937005145475 0.0003191372630967521
    5001 0.9918764412868768 0.0003189294427111535
    5101 1.2255524442298338 0.0003187220277918973
    5201 1.3292681298672733 0.0003185150170222263
    5301 0.878005885053426 0.00031830840909136197
    5401 1.0740752452165907 0.00031810220269447
    5501 1.0994656120092259 0.00031789639653262544
    5601 1.159670107124839 0.0003176909893127784
    5701 0.8859041188843548 0.0003174859797477199
    5801 1.084522244927939 0.00031728136655604814
    5901 1.4824702723776682 0.0003170771484621346
    6001 1.230977819112013 0.0003168733241960908
    6101 1.0119965468620649 0.00031666989249373517
    6201 1.1002646164814678 0.00031646685209656003
    6301 1.1440792203939054 0.00031626420175169897
    1 1.0551144047021808 0.0003160882122689669
    101 1.0408390048833098 0.00031588628797931984
    201 1.0153360446565785 0.0003156847501772966
    301 1.01748421555385 0.0003154835976315582
    401 1.1267781729111448 0.0003152828291162512
    501 1.0327468327741371 0.0003150824434109757
    601 0.9960314880299848 0.0003148824393007546
    701 1.0631007702104398 0.00031468281557600267
    801 1.0372768385277595 0.00031448357103249544
    901 1.031023440795252 0.0003142847044713392
    1001 0.9323838343843818 0.0003140862146989404
    1101 0.850510573014617 0.0003138881005269756
    1201 1.0943628003296908 0.0003136903607723615
    1301 1.0844742289336864 0.00031349299425722566
    1401 1.4024500491796061 0.00031329599980887637
    1501 1.1463393379817717 0.00031309937625977405
    1601 1.0650637014914537 0.0003129031224475018
    1701 1.1410465109511279 0.00031270723721473664
    1801 1.0148204645956866 0.0003125117194092209
    1901 0.9743651752360165 0.0003123165678837336
    2001 0.990701739974611 0.00031212178149606226
    2101 1.1128740338463103 0.00031192735910897496
    2201 1.5508626039809315 0.0003117332995901923
    2301 1.01486110695987 0.00031153960181235955
    2401 0.9611194784665713 0.0003113462646530196
    2501 1.0847897573257796 0.0003111532869945851
    2601 1.2803321699175285 0.00031096066772431187
    2701 1.0769891024538083 0.0003107684057342714
    2801 1.0039243546780199 0.00031057649992132457
    2901 1.0824949400266632 0.00031038494918709473
    3001 0.9644128995714709 0.00031019375243794144
    3101 0.9868561172188492 0.00031000290858493437
    3201 1.0903575613629073 0.00030981241654382685
    3301 1.2633273452374851 0.0003096222752350304
    3401 1.0088038056419464 0.000309432483583589
    3501 1.1047235757578164 0.0003092430405191533
    3601 1.163446888080216 0.00030905394497595545
    3701 1.018205283649877 0.00030886519589278384
    3801 1.0408570388099179 0.00030867679221295824
    3901 1.0657447287230752 0.00030848873288430483
    4001 0.930832964291767 0.0003083010168591314
    4101 1.4587874389944773 0.00030811364309420327
    4201 1.1227497690124437 0.0003079266105507184
    4301 1.1970081577601377 0.0003077399181942835
    4401 1.1083186157047749 0.00030755356499488986
    4501 1.0473160178516991 0.00030736754992688985
    4601 1.0928417469840497 0.0003071818719689727
    4701 1.0073067757184617 0.00030699653010414117
    4801 1.1523846584532293 0.00030681152331968824
    4901 1.1988015054084826 0.0003066268506071739
    5001 1.036811558995396 0.00030644251096240176
    5101 1.0675939484208357 0.0003062585033853964
    5201 1.0752534797684348 0.00030607482688038056
    5301 1.1183025861246279 0.0003058914804557523
    5401 1.0365500289481133 0.0003057084631240629
    5501 1.129350705537945 0.00030552577390199393
    5601 1.06671357084997 0.0003053434118103358
    5701 1.0859052878222428 0.0003051613758739652
    5801 1.0723684425465763 0.00030497966512182316
    5901 1.0603833084605867 0.0003047982785868937
    6001 1.0581669194652932 0.0003046172153061818
    6101 1.2675022858026068 0.0003044364743206923
    6201 1.0536423055746127 0.0003042560546754083
    6301 1.2465427681345318 0.00030407595541926996
    1 0.8441335130482912 0.00030391413923858634
    101 0.9350836968515068 0.00030373464611573535
    201 1.0421846130630001 0.0003035554706461405
    301 1.008769184758421 0.0003033766118939749
    401 0.9787611065548845 0.000303198068927267
    501 1.0469113910803571 0.00030301984081788013
    601 1.0306272330635693 0.00030284192664149214
    701 0.8391264111269265 0.00030266432547757535
    801 0.9852646276758605 0.00030248703640937665
    901 0.9640902730752714 0.00030231005852389745
    1001 1.1280414578068303 0.00030213339091187405
    1101 0.9939612101297826 0.0003019570326677579
    1201 1.0867023059399799 0.00030178098288969626
    1301 1.0411206257122103 0.00030160524067951265
    1401 1.0711584523378406 0.0003014298051426879
    1501 1.0796883448783774 0.0003012546753883405
    1601 1.027666941517964 0.00030107985052920836
    1701 1.082986782770604 0.00030090532968162913
    1801 1.1074479344606516 0.0003007311119655219
    1901 1.1305997944582487 0.0003005571965043686
    2001 1.345339710366943 0.0003003835824251953
    2101 1.001590578132891 0.00030021026885855383
    2201 1.0054450589232147 0.0003000372549385033
    2301 1.0041432639409322 0.00029986453980259265
    2401 1.0640304164699046 0.00029969212259184163
    2501 1.1201181028736755 0.0002995200024507235
    2601 0.8765224074013531 0.00029934817852714696
    2701 1.0152945614827331 0.0002991766499724386
    2801 1.2956223709957158 0.0002990054159413251
    2901 1.097986907014274 0.0002988344755919157
    3001 1.2291080165232415 0.00029866382808568526
    3101 1.1140619127836544 0.0002984934725874564
    3201 1.0722678579004423 0.0002983234082653825
    3301 1.03946516571159 0.0002981536342909311
    3401 0.9876702070032479 0.0002979841498388662
    3501 0.8540887358831242 0.00029781495408723205
    3601 0.9894529741141014 0.00029764604621733594
    3701 1.0710785342380404 0.000297477425413732
    3801 1.2710673977526312 0.00029730909086420423
    3901 1.0929949116252828 0.0002971410417597504
    4001 1.1222996068827342 0.0002969732772945655
    4101 1.164539644116303 0.00029680579666602566
    4201 1.033734397671651 0.000296638599074672
    4301 1.093015514779836 0.0002964716837241944
    4401 1.0481613585725427 0.0002963050498214161
    4501 1.0937337041832507 0.00029613869657627706
    4601 1.1815662420112858 0.000295972623201819
    4701 1.1428916620570817 0.0002958068289141693
    4801 1.1009264337189961 0.0002956413129325257
    4901 0.8777621657354757 0.00029547607447914055
    5001 1.0156730558082927 0.00029531111277930595
    5101 0.9942513670539483 0.00029514642706133804
    5201 1.1302866424011881 0.00029498201655656206
    5301 1.0087051462905947 0.0002948178804992971
    5401 1.0660143050336046 0.0002946540181268415
    5501 1.0107977577135898 0.00029449042867945755
    5601 1.0777363086963305 0.0002943271114003569
    5701 0.856828257907182 0.00029416406553568584
    5801 1.1287360956775956 0.0002940012903345107
    5901 1.366358119645156 0.00029383878504880313
    6001 1.0964845723065082 0.00029367654893342604
    6101 1.34646458978159 0.00029351458124611887
    6201 1.2197050878312439 0.0002933528812474836
    6301 1.0830936049751472 0.0002931914482009704
    1 1.2537849597129025 0.00029304477550482497
    101 0.9655292083625682 0.00029288385030021
    201 1.1372923650196753 0.0002927231899204302
    301 0.9451354363700375 0.0002925627936399378
    401 0.9661671929707154 0.00029240266073596516
    501 1.0162687979172915 0.0002922427904885108
    601 0.9551542007829994 0.0002920831821803257
    701 0.9841917234880384 0.0002919238350969
    801 1.061543255826109 0.00029176474852644945
    901 0.985908080736408 0.0002916059217599022
    1001 1.0337736615701942 0.0002914473540908853
    1101 0.9899544979416532 0.0002912890448157118
    1201 1.1052642236463726 0.00029113099323336726
    1301 0.9609193275682628 0.00029097319864549706
    1401 1.0143777604680508 0.0002908156603563932
    1501 1.0131031578639522 0.0002906583776729816
    1601 1.0399196342332289 0.00029050134990480915
    1701 1.3567781529854983 0.00029034457636403104
    1801 1.1145936762022757 0.0002901880563653981
    1901 1.0984270876506343 0.0002900317892262443
    2001 1.0226630433771788 0.00028987577426647405
    2101 1.1856945900362916 0.0002897200108085499
    2201 1.125245438015554 0.00028956449817748025
    2301 1.126162831991678 0.00028940923570080693
    2401 0.971063018507266 0.00028925422270859307
    2501 0.8773902256507427 0.00028909945853341086
    2601 0.8763325407635421 0.0002889449425103295
    2701 1.1415061227562546 0.0002887906739769035
    2801 1.052460735765635 0.0002886366522731603
    2901 1.5205009760629764 0.00028848287674158846
    3001 0.9414957111439435 0.0002883293467271265
    3101 0.9435001520323567 0.0002881760615771502
    3201 1.1403545759221743 0.0002880230206414618
    3301 1.053262686386006 0.00028787022327227786
    3401 1.0832857484929264 0.000287717668824218
    3501 1.1019753235159442 0.00028756535665429354
    3601 1.055358653771691 0.0002874132861218958
    3701 1.0278627741499804 0.00028726145658878504
    3801 1.0083879251906183 0.0002871098674190792
    3901 1.0555589499126654 0.0002869585179792425
    4001 1.0509156602493022 0.00028680740763807453
    4101 1.0385481148259714 0.0002866565357666993
    4201 1.0417402729653986 0.0002865059017385537
    4301 1.082753369351849 0.0002863555049293774
    4401 1.0459589868987678 0.0002862053447172013
    4501 1.0592919351911405 0.00028605542048233684
    4601 1.1034397517796606 0.0002859057316073656
    4701 1.1311876591207692 0.00028575627747712837
    4801 1.070465801298269 0.00028560705747871445
    4901 1.199700104945805 0.00028545807100145134
    5001 1.2379299406893551 0.00028530931743689397
    5101 1.1045849512156565 0.00028516079617881457
    5201 0.9958119990806154 0.000285012506623192
    5301 1.620139996672151 0.00028486444816820157
    5401 1.0613090786646353 0.0002847166202142048
    5501 1.175794189737644 0.0002845690221637393
    5601 1.1241089710965753 0.00028442165342150834
    5701 1.160597581154434 0.000284274513394371
    5801 1.0310369414401066 0.0002841276014913322
    5901 0.9960121748881647 0.0002839809171235324
    6001 0.9698299318188219 0.00028383445970423817
    6101 1.1415085992775857 0.0002836882286488319
    6201 1.0641657677479088 0.0002835422233748022
    6301 1.0828722650112468 0.00028339644330173413



```python
1 10.825187489390373 6.987712429686844e-07
101 9.447168171405792 3.56373333914029e-05
201 7.142856806516647 7.057589553983712e-05
301 6.237934365868568 0.00010551445768827134
401 5.762486848048866 0.00014045301983670557
501 5.415792358107865 0.00017539158198513977
601 5.081815680023283 0.000210330144133574
701 4.788327748770826 0.00024526870628200823
801 4.381739928154275 0.0002802072684304424
901 4.55433791608084 0.00031514583057887664
1001 4.911875109748507 0.0003500843927273108
1101 4.0579032292589545 0.0003850229548757451
1201 4.2276234351193125 0.0004199615170241793
1301 3.932735869428143 0.00045490007917261356
1401 3.8179439397063106 0.0004898386413210477
1501 3.3608515430241823 0.000524777203469482
1601 3.832796103321016 0.0005597157656179162
1701 2.907085266895592 0.0005946543277663504
1801 3.5280659823838505 0.0006295928899147847
1901 2.895841649500653 0.0006645314520632189
2001 3.273784235585481 0.000699470014211653
2101 3.181488689899197 0.0007344085763600873
2201 3.4151616653980454 0.0007693471385085215
2301 3.4343731447588652 0.0008042857006569557
2401 3.0505455391539726 0.0008392242628053899
2501 2.8089329147478566 0.0008741628249538242
2601 2.7827929875456903 0.0009091013871022583
2701 2.4428516102489084 0.0009440399492506926
2801 2.4015486147254705 0.0009789785113991267
2901 2.3568112018401735 0.001013917073547561
3001 2.6349758653668687 0.0010488556356959952
3101 2.5981983028614195 0.0010837941978444295
3201 2.666826274838968 0.0011187327599928637
3301 3.0092043554177508 0.0011536713221412978
3401 2.4580375660589198 0.0011886098842897321
3501 2.586465588421561 0.0012235484464381662
3601 2.5663993963389657 0.0012584870085866006
3701 2.9430236657499336 0.0012934255707350347
3801 2.464644919440616 0.001328364132883469
3901 2.7124062888276512 0.0013633026950319032
4001 2.646443709731102 0.0013971932312809247
4101 2.7294750874862075 0.001380057517579748
4201 2.1295202329056337 0.0013635372009002666
4301 2.596563663915731 0.001347596306985731
4401 2.1265982036820787 0.0013322017384983986
4501 2.3880532500334084 0.0013173229858148
4601 2.6129120760888327 0.0013029318725783852
4701 2.2873719420749694 0.001289002331178292
4801 2.4949760700110346 0.0012755102040816328
4901 2.496607314562425 0.001262433067573089
5001 2.1889712483389303 0.0012497500749750088
5101 1.8677761815488338 0.0012374418168536253
5201 2.2992054556962103 0.0012254901960784316
5301 2.664361578106707 0.0012138783159049418
5401 2.705850490485318 0.0012025903795063202
5501 2.581445264921058 0.0011916115995949978
5601 2.2480602325085783 0.0011809281169581616
5701 1.9289666265249252 0.0011705269268863989
5801 2.4863578918157145 0.0011603958126073107
5901 2.632946971571073 0.0011505232849492607
6001 2.496141305891797 0.0011408985275576757
6101 2.6422974687084206 0.0011315113470699342
6201 2.448802186456305 0.0011223521277270118
```