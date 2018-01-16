from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils
from scipy import ndimage
from matplotlib import pyplot as plt
from myutilities import *

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--preprocess", required=True,
	help="Snapped sudoku")
ap.add_argument("-i", "--image", required = True,
	help = "Path to the scanned image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
preproc = args["preprocess"]

case = "True"

if preproc == str(2):
	case = "False"

th3,warped1,warped = preprocess(image,case)
warped2 = warped1.copy()
img = grids(warped,warped2)
c2,bm,cnts = grid_points(img,warped2)
c2,num,cx,cy = get_digit(c2,bm,warped1,cnts)
grid = sudoku_matrix(num)

if(sudoku(grid)):
	arr = board(grid)
	overlay(arr,num,warped1,cx,cy)

else:
	print("There is no solution")


