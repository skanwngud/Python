import numpy as np
import tensorflow as tf

from tensorflow.contrib.training import HParams

# parameters 
def default_hparams():
    return HParams(
        n_vocab = 0,
        n_ctx = 1024,
        n_embd = 768,
        n_head = 12,
        n_layer = 12
    )

def shape_list(x):
    static = x.shape.as_list() # 정적 shape
    dynamic = tf.shape(x)      # 동적 shape
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

# define softmax
def softmax(x, axis = 1):
    x = x - tf.reduce_max(x, axis = axis, keepdims = True)      # keepdims : 행렬의 합산 후에도 차원을 유지한다
    ex = tf.exp(x)
    return ex / tf.reduce_sum(ex, axis = axis, keepdims = True)

# define gelu
def gelu(x):
    return 0.5 * x * (1 + tf.tanh(np.sqrt(2/np.pi) * (x+0.044715 * tf.pow(x, 3))))
"""
NLP 에서 주로 사용 되는 ELU 계열의 활성화 함수
0 미만인 값에서 전부 0 으로 처리하는 RELU 와
0 미만인 값에서 -1 로 수렴 되는 ELU 와 달리,
0 미만인 값에서 0 으로 수렴 되기는 하지만, 특정 구간 사이에서 약간의 기울기를 준다
"""

# normalization
def norm(x, scope, *, axis = -1, epsilon = 1e-5):
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

def split_states(x, n):
    *start, m = shape_list(x)
    return tf.reshape(x, start + [n, m//n])

"""
*start : start 라는 변수에 여러개의 값을 저장할 때 사용
"""

def merge_states(x):
    *start, a, b = shape_list(x)
    return tf.reshape(x, start + [a*b])

# conv1d layer 선언
def conv1d(x, scope, nf, *, w_init_stdev = 0.02):
    with tf.variable_scope(scope):
        *start, nx = shape_list(x)
        w = tf.get_variable('w', [1, nx, nf], initializer=tf.random_normal_initializer(stddev=w_init_stdev))
        b = tf.get_variable('b', [nf], initializer=tf.constant_initializer(0))
        c = tf.reshape(tf.matmul(tf.reshape(x, [-1, nx]), tf.reshape(w, [-1, nf]))+b, start+[nf])
        return c

