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
	rand = random.randint(0,1)
	dim = im1.shape[rand]

	randSplit = random.randint(0, dim)

	if rand == 0:
		im1[randSplit:dim, :, :] = im2[randSplit:dim, :, :]
	else:
		im1[:, randSplit:dim, :] = im2[:, randSplit:dim, :]
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
	newcolor = [random.randint(0,255), random.randint(0,255), random.randint(0,255)]
	# print(newcolor)

	flatImage = im.reshape(im.shape[0] * im.shape[1], 3)
	colors = [list(x) for x in flatImage]
	colors = np.unique(colors, axis=0)
	
	i = random.randint(0, colors.shape[0]-1)

	# print(colors[i])

	# x, y, z = np.where(im == colors[i])

	newImage = np.copy(im)

	M, N, C = im.shape

	for k in range(M):
		for j in range(N):
			if np.array_equal(newImage[k][j], colors[i]):
				newImage[k][j] = newcolor

	# newImage[x[0]:x[-1]+1, y[0]:y[-1]+1] = newcolor

	# plt.imshow(newImage)
	# plt.waitforbuttonpress(0)

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

	return colors.shape[0]

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
	plt.imsave("red.tiff", red/255)

	blue = np.zeros((400,800,3))
	blue[:,:,2] = 255
	# uncomment the lines below to view the image

	# im1 = recombine(red, blue)
	# im2 = recombine(blue, red)

	# for i in range(0, 20):
	# 	im1 = mutate(recombine(im1, im2))
	# 	im2 = mutate(recombine(im2, im1))

	# im1 = mutate(recombine(im1, im2))
	# print(evaluate(im1))
	# plt.imshow(im1)
	# plt.show() 

	
if __name__ == '__main__':
	main()

