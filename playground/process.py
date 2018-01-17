import time
import threading
import multiprocessing
from multiprocessing import Process
import argparse
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt

class Data():

	def __init__(self):
		self._x = 0

	@property
	def x(self):
		return self._x

	@x.setter
	def x(self, v):
		self._x += v

def inc(v):

	print('Processor = {}, v = {}'.format(multiprocessing.current_process().name, v))
	def fct(d):
		d.x = v
		print('x = {}'.format(d.x))

	return fct

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

def test_process():
	print('Numbar of CPUs = {}.'.format(multiprocessing.cpu_count()))


	d = Data()
	ps = [Process(target=inc(_+1), args=(d, ), name=_) for _ in range(5)]
	for p in ps:
		p.start()

	for p in ps:
		p.join()

	print('x = {}'.format(d.x))

def test_thread():
	pass


def test_setgetter():
	d = Data()

	for i in range(2,6):
		d.x = i
		print(d.x)

if __name__ == '__main__':
	# main()
	# test_setgetter()
	test_process()