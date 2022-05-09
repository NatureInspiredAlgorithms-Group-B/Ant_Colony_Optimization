import random
import numpy as np
from graph import *



if __name__ == '__main__':
	G = TSP()
	N = G[0]
	N.value = 1
	E = G[0, 1]

	print(N)
	print(E)
	print(E.value)
