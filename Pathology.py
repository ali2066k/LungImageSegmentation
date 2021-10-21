import random
import cv2
import numpy as np

def intersect(img, thresh):
	tab = np.array(thresh)
	image = np.array(img)
	return np.where(image==tab, image, 0)
	
def add_noise(img, medval, img_rows=256, img_cols=256):
	variation = np.random.randint(-15,15, size=(img_rows,img_cols))
	tab = np.array(img)
	noise = np.zeros((img_rows,img_cols))
	a = np.where(tab != 255, noise, variation + medval)
	a = np.where(a < 255, a, 255)
	a = np.where(a > 0, a, 0)
	return a
	
def imagefinal(img,lung):
	current = np.array(img)
	if len(lung.shape) >= 3 :
		original=np.array(lung[:, :, 0])
	else:
		original=np.array(lung)
	return np.where(current<original, original, current)

def createCircle(img,nb):
	for i in range(nb):
		posX= random.randint(100,412)
		posY = random.randint(100, 412)
		rad = random.randint(20,70)
		circle = cv2.circle(img,(posX,posY),radius=rad,color=255,thickness=-1)


def createFakeImage(lung,mask,minval,maxval, img_rows=256, img_cols=256):
	img = np.zeros((img_rows, img_cols))
	createCircle(img, 20)
	thresh = np.zeros((img_rows, img_cols))
	ret, thresh = cv2.threshold(mask, 5, 255, cv2.THRESH_BINARY)
	image = intersect(img,thresh)
	medval = random.randint(minval, maxval)
	img = add_noise(image,medval, img_rows, img_cols)
	lung = imagefinal(img, lung)
	image = np.array(lung)
	return image

def main ():
	lung= cv2.imread('lung.pgm')
	mask= cv2.imread('msk.pgm')
	#mask = mask[:,:,0]
	minval=70
	maxval=230
	
	createFakeImage(lung,mask,minval,maxval)
	return

if __name__ == '__main__':
	main()



