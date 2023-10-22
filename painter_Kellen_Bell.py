import skimage.io as io
import numpy as np
from matplotlib import pyplot as plt
import random
import argparse


def recombine(
	im1: np.ndarray, im2: np.ndarray
) -> np.ndarray:
	"""Create a new image from two images. 

	Vars:
		im1: the first image
		im2: the second image

	Returns:
		A new image, chosen by first, randomly choosing
		between the horizontal or vertical orientation,
		and then slicing each image into two pieces along
		a randomly-chosen vertical or horizontal line.
	"""
	# random number 1 or 0
	rand = random.randint(0,1)

	# dimensions either length or width
	dim = im1.shape[rand]

	# find a random line to split on
	randSplit = random.randint(0, dim)

	# split horiz if 0
	if rand == 0:
		im1[randSplit:dim, :, :] = im2[randSplit:dim, :, :]

	# split vert
	else:
		im1[:, randSplit:dim, :] = im2[:, randSplit:dim, :]

	# return the combined image
	return im1


def mutate(im: np.ndarray) -> np.ndarray:
	"""Mutate an image.

	Vars:
		im: the image to mutate.

	Returns:
		A new image, which is the same as the original,
		except that on of the colors is the image is
		globally (i.e., everywhere it occurs in the image)
		replace with a randomly chosen new color.
	"""
	im = im.astype(int)

	# make a new color randomly
	newcolor = [random.randint(0,255), random.randint(0,255), random.randint(0,255)]

	# find all the unique colors in the image
	flatImage = im.reshape(im.shape[0] * im.shape[1], 3)
	colors = [list(x) for x in flatImage]
	colors = np.unique(colors, axis=0)
	
	# pick a color to replace
	i = random.randint(0, colors.shape[0]-1)

	newImage = np.copy(im)

	M, N, C = im.shape

	# check for the color and replace it
	for k in range(M):
		for j in range(N):
			if np.array_equal(newImage[k][j], colors[i]):
				newImage[k][j] = newcolor

	return newImage



def evaluate(im: np.ndarray):
	"""Evaluate an image.

	Vars:
		im: the image to evaluate.

	Returns:
		The value of the evaluation function on im.
		Since art is subjective, you have complete
		freedom to implement this however you like.
	"""
	# select for more more colors
	flatImage = im.reshape(im.shape[0] * im.shape[1], 3)
	colors = [list(x) for x in flatImage]
	colors = np.unique(colors, axis=0)

	# select for more horiz lines

	# offset horizontally
	horiz_offset = np.zeros_like(im)
	for i in range(horiz_offset.shape[0]):
		for j in range(horiz_offset.shape[1]):
			if i < im.shape[0] - 1:
				horiz_offset[i][j] = im[i+1][j]

	num_lines = 0
	# find the difference
	horiz_lines = im - horiz_offset

	# check for horiz lines
	for i in range(horiz_lines.shape[0]):
		for j in range(horiz_lines.shape[1]):
			if all(horiz_lines[i][j]) != all([0, 0, 0]):
				num_lines += 1

	# penalize black colors
	x, y, z = np.where(im == [0, 0, 0])
	count_black = len(x)

	# points are num colors + horiz_lines - amount of black
	return colors.shape[0] + (num_lines // horiz_lines.shape[1]) - (count_black // im.shape[1])

def main():
	parser = argparse.ArgumentParser(
    	prog='painter',
    	description='creates paintings according to a genetic algorithm'
	)

	parser.add_argument('-g', '--generations', default=100, help="The number of generations to run", type=int)
	parser.add_argument('-p', '--pools', default=10, help="The size of the pool", type=int)
	parser.add_argument('-m', '--mutation', default=.2, help="The chance of a mutation", type=float)
	parser.add_argument('-r', '--recombine', default = 2, help="The number of pairs to recombine in each generation", type=int)
	args = parser.parse_args()
	
	red = np.zeros((400,800,3))
	red[:,:,0] = 255

	blue = np.zeros((400,800,3))
	blue[:,:,2] = 255

	# init the pool with red and blue images
	initPool = [0] * args.pools
	for i in range(args.pools):
		if i % 2 == 0:
			initPool[i] = [np.copy(red), 0]
		else:
			initPool[i] = [np.copy(blue), 0]

	# loop through each generation
	for i in range(args.generations):

		# eval images and sort by points
		for l in range(args.pools):
			initPool[l][1] = evaluate(initPool[l][0])
		initPool = sorted(initPool, key=lambda x: x[:][1], reverse=True)

		# recombine the top pairs
		for k in range(args.recombine):
			if k < args.pools - 1:
				initPool[k][0] = recombine(initPool[k][0], initPool[k+1][0])

		# loop through each image with a chance of mutation
		for j in range(args.pools):
			mutateChance = random.uniform(0,1)
			if mutateChance <= args.mutation:
				initPool[j][0] = mutate(initPool[j][0])

	# sort one last time
	for l in range(args.pools):
		initPool[l][1] = evaluate(initPool[l][0])
	initPool = sorted(initPool, key=lambda x: x[:][1], reverse=True)

	# output top 3 images
	for i in range(0, 3):
		plt.imsave("art{}.tiff".format(i+1), initPool[i][0]/255)

	
if __name__ == '__main__':
	main()

