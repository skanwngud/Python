import numpy as np
import tensorflow as tf

from tensorflow.contrib.training import HParams

# parameters
def default_hparams(): # interatcive_conditional_samples.py 에서 사용 
    return HParams(
        n_vocab = 0,
        n_ctx = 1024,
        n_embd = 768,
        n_head = 12,
        n_layer = 12
    )

def shape_list(x): # split_state, merge_state, conv1d, attn/mask_attn_weights, model 에서 사용
    static = x.shape.as_list() # list 형태로 shape 출력
    dynamic = tf.shape(x)      
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

# define softmax
def softmax(x, axis = 1): # attn/multi_head_attn 에서 사용
    x = x - tf.reduce_max(x, axis = axis, keepdims = True)      # keepdims : 행렬의 합산 후에도 차원을 유지한다
    ex = tf.exp(x)
    return ex / tf.reduce_sum(ex, axis = axis, keepdims = True)

# define gelu
def gelu(x): # mlp 에서 사용
    return 0.5 * x * (1 + tf.tanh(np.sqrt(2/np.pi) * (x+0.044715 * tf.pow(x, 3))))
"""
NLP 에서 주로 사용 되는 ELU 계열의 활성화 함수
0 미만인 값에서 전부 0 으로 처리하는 RELU 와
0 미만인 값에서 -1 로 수렴 되는 ELU 와 달리,
0 미만인 값에서 0 으로 수렴 되기는 하지만, 특정 구간 사이에서 약간의 기울기를 준다
"""

# normalization
def norm(x, scope, *, axis = -1, epsilon = 1e-5): # block, model 에서 사용
    with tf.variable_scope(scope):
        n_state = x.shape[-1].value
        g = tf.get_variable('g', [n_state], initializer=tf.constant_initializer(1))
        b = tf.get_variable('b', [n_state], initializer=tf.constant_initializer(0))
        u = tf.reduce_mean(x, axis = axis, keepdims = True)
        s = tf.reduce_mean(tf.square(x-u), axis = axis, keepdims=True)
        x = (x - u) * tf.rsqrt(s + epsilon)
        x = x*g + b
        return x
"""
scope : 변수의 범위를 지정함
tf.get_variable() 에 전달 된 이름의 namespace 들을 관리함

tf.constant_initializer() : 제공된 값으로 모든 것을 초기화 함
tf.rsqrt : sqrt 에 제곱근을 씌움
"""

def split_states(x, n): # attn/split_head
    *start, m = shape_list(x)
    return tf.reshape(x, start + [n, m//n])

"""
*start : start 라는 변수에 여러개의 값을 저장할 때 사용
"""

def merge_states(x): # attn/merge_head
    *start, a, b = shape_list(x)
    return tf.reshape(x, start + [a*b])

# conv1d layer 선언
def conv1d(x, scope, nf, *, w_init_stdev = 0.02): # attn, mlp
    with tf.variable_scope(scope):
        *start, nx = shape_list(x)
        w = tf.get_variable('w', [1, nx, nf], initializer=tf.random_normal_initializer(stddev=w_init_stdev))
        b = tf.get_variable('b', [nf], initializer=tf.constant_initializer(0))
        c = tf.reshape(tf.matmul(tf.reshape(x, [-1, nx]), tf.reshape(w, [-1, nf]))+b, start+[nf])
        return c
"""
nx = input_dims
nf = output_dims
"""


def attention_mask(nd, ns, *, dtype): # attn/mask_attn_weights
    i = tf.range(nd)[:, None]
    j = tf.range(ns)
    m = i >= j - ns + nd
    return tf.cast(m, dtype)

"""
tf.cast : 텐서를 새로운 형태로 casting 하는데 사용
부동소수점형에서 정수형으로 바꾸는 경우 소수점을 버린다
"""

def attn(x, scope, n_state, *, past, hparams): # block
    assert x.shape.ndims == 3
    assert n_state % hparams.n_head == 0
    """
    assert(가정설정문) : 조건, 메세지(메세지는 생략 가능)
    x.shape.ndims == 3 인 경우만 True, 그 외는 전부 예외처리함

    ndims = number of dimensions (차원의 수)

    """
    if past is not None:
        assert past.shape.ndims == 5

    def split_heads(x):
        return tf.transpose(split_states(x, hparams.n_head), [0, 2, 1, 3])

    def merge_head(x):
        return merge_states(tf.transpose(x, [0, 2, 1, 3]))

    def mask_attn_weights(w):
        _, _, nd, ns = shape_list(w)
        b = attention_mask(nd, ns, dtype = w.dtype)
        b = tf.reshape(b, [1, 1, nd, ns])
        w = w*b - tf.cast(1e10, w.dtype)*(1-b)
        return w

    def multihead_attn(q, k, v):
        w = tf.matmul(q, k, transpose_b=True)
        w = w * tf.rsqrt(tf.cast(v.shape[-1].value, w.dtype))
        """
        transpose_b = True : 행렬곱을 할 때 shape 를 맞춰주기 위해 전치시킴
        """

        w = mask_attn_weights(w)
        w = softmax(w)
        a = tf.matmul(w, v)
        return a

    with tf.variable_scope(scope):
        c = conv1d(x, 'c_attn', n_state * 3)
        q, k, v = map(split_heads, tf.split(c, 3, axis = 2))
        present = tf.stack([k, v], axis = 1)
        if past is not None:
            pk, pv = tf.unstack(past, axis = 1)
            k = tf.concat([pk, k], axis = -2)
            v = tf.concat([pv, v], axis = -2)
        a = multihead_attn(q, k, v)
        a = merge_head(a)
        a = conv1d(a, 'c_proj', n_state)
        return a, present

def mlp(x, scope, n_state, *, hparams): # block
    with tf.variable_scope(scope):
        nx = x.shape[-1].value
        h = gelu(conv1d(x, 'c_fc', n_state))
        h2 = conv1d(h, 'c_proj', nx)
        return h2

def block(x, scope, *, past, hparams): # model
    with tf.variable_scope(scope):
        nx = x.shape[-1].value
        a, present = attn(norm(x, 'ln_1'), 'attn', nx, past=past, hparams=hparams)
        x += a
        m = mlp(norm(x, 'ln_2'), 'mlp', nx * 4, hparams=hparams)
        x += m
        return x, present

def past_shape(*, hparams, batch_size = None, sequence = None):
    return [batch_size, hparams.n_layer, 2, hparams.n_head, sequence, hparams.n_embd//hparams.n_head]

def expand_tile(value, size): # position_for
    value = tf.convert_to_tensor(value, name= 'value')
    ndims = value.shape.ndims
    return tf.tile(tf.expand_dims(value, axis = 0), [size] + [1]*ndims)

def position_for(tokens, past_length): # model
    batch_size = tf.shape(tokens)[0]
    nsteps = tf.shape(tokens)[1]
    return expand_tile(past_length + tf.range(nsteps), batch_size)

def model(hparams, X, past = None, scope = 'model', reuse = False):
    with tf.variable_scope(scope, reuse = reuse):
        results = {}
        batch, sequence = shape_list(X)

        wpe = tf.get_variable('wpe', [hparams.n_ctx, hparams.n_embd],
                                initializer = tf.random_normal_initializer(stddev = 0.01))
        wte = tf.get_variable('wte', [hparams.n_vocab, hparams.n_embd],
                                initializer =tf.random_normal_initializer(stddev = 0.02))
        past_length = 0 if past is None else tf.shape(past)[-2]
        h = tf.gather(wte, X) + tf.gather(wpe, position_for(X, past_length))

        presents = list()
        pasts = tf.unstack(past, axis = 1) if past is not None else [None] * hparams.n_layer
        assert len(pasts) == hparams.n_layer
        for layer, past in enumerate(pasts):
            h, present = block(h, 'h%d' % layer, past = past, hparams = hparams)
            presents.append(present)
        results['present'] = tf.stack(presents, axis = 1)
        h = norm(h, 'ln_f')

        h_flat = tf.reshape(h, [batch * sequence, hparams.n_embd])
        logits = tf.matmul(h_flat, wte, transpose_b = True)
        logits = tf.reshape(logits, [batch, sequence, hparams.n_vocab])
        results['logtis'] = logits
        return results