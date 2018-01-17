# test_softmax.py
import tensorflow as tf
import sys
import argparse
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from getchar import _Getch

def test_softmax():
    nlog = 3
    logits = tf.placeholder(dtype=tf.float32, shape=(None, nlog))
    advantage = tf.placeholder(dtype=tf.float32, shape=(None, nlog))
    
    ce = tf.nn.softmax_cross_entropy_with_logits(labels=advantage, logits=logits)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    feed_dict = {
        logits: np.array([[1/3, 1/3, 1/3], [0.98, 0.01, 0.01]]), 
        advantage: np.array([[10, 0, 0], [-10, 0, 0]])
    }
    res = sess.run(ce, feed_dict=feed_dict)
    print(res)
  
if __name__ == '__main__':
    test_softmax()

