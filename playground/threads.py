import gym
import tensorflow as tf

import time
import threading
import argparse
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt

class Agent(threading.Thread):
	def __init__(self, i):
		threading.Thread.__init__(self)
		self.i = i

	def run(self):
		for _ in range(4):
			time.sleep(np.random.uniform(0.0, 1.0))
			print('Thread {} : {}'.format(self.i, (_+1)*self.i))

def main():
	agents = [Agent(i) for i in range(4)]

	for agent in agents:
		agent.start()

	for agent in agents:
		agent.join()

	print('All done')

if __name__ == '__main__':
	main()