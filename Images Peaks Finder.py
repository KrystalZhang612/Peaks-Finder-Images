
import numpy as np

import matplotlib.image as mpimg

import matplotlib.pyplot as plt

from PIL import Image

import cv2

import math




#we here have alternative ways to implement SobelImage and BilinearInterpolation

#as two supplemental functions to implement FindPeaksImage function 


def SobelImage(image):
	
	#open the image and convert it to greyscale 
	
	image1 = Image.open(image).convert('LA')
	
	Width = image1.width
	
	Height = image1.height
	
	#Load the numpy array of a single sample image
	
	Pixel = image1.load()
	
	#arranging RGB colors for the grayscale image
	
	def RBGFORGRAYSCALE(rgb):
		
		#Luminance percieved 
		
		return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
	
	#replace imread(image) to imageio.imread(image)
	
	TheImage = mpimg.imread(image)
	
	GrayScale = RBGFORGRAYSCALE(TheImage)
	
	GrayScale = GrayScale * 255
	
	
	#creating gradient filter
	
	
	GradientX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
	
	GradientY = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
	
	factor = 3 
	
	Magnitude = np.zeros((Height, Width), dtype=np.float32)
	
	Orientation = np.zeros((Height, Width), dtype=np.float32)
	
	#initialize an empty 2D array for orientation 
	
	
	OrtArray = np.zeros((Height, Width, 3), dtype=np.float32)
	
	CurrentPixel = GrayScale
	
	#similar way of creating the filter 
	
	for i in range(1, Height - 1):
		
		for j in range(1, Width - 1):
			
			#current position coordinate 
			
			CurrentPosition = (i, j)
			
			
			#temp holder 
			
			Temp = CurrentPixel[CurrentPosition[0] - (factor // 2): CurrentPosition[0] + (factor // 2) + 1, 
				
				CurrentPosition[1] - (factor // 2):CurrentPosition[1] + (factor// 2) + 1]
			
			
			#Arranging the horizontal and vertical axises 
			
			Temp1 = Temp * GradientX
			
			Temp1 = Temp1.sum(axis=0)
			
			Temp1 = Temp1.sum(axis=0)
			
			Temp2 = Temp * GradientY
			
			Temp2 = Temp2.sum(axis=0)
			
			Temp2 = Temp2.sum(axis=0)
			
			
			Magnitude[i, j] = (Temp2 ** 2 + Temp1 ** 2) ** 0.5
			
			#use math.atan(float) to return the arc tangent of different numbers
			
			if Temp1 == 0 and Temp2 > 0:
				
				Orientation[i, j] = math.degrees(math.atan(float('inf')))
				
				
			elif Temp1 == 0 and Temp2 < 0:
				
				Orientation[i, j] = math.degrees(math.atan(float('-inf')))
				
			elif Temp1 == 0 and Temp2 == 0:
				
				Orientation[i, j] = 1
				
			elif Temp1 != 0:
				
				Orientation[i, j] = (math.degrees(math.atan(Temp2 / Temp1)))
				
			else:
				
				#Temp1 can neither be greater nor samller than 0 here 
				
				print("ERROR! ")
				
				#followed by the limits of gradient magnitude
				
	Magnitude = Magnitude * 255 / Magnitude.max()
	
	
	
	
	for i in range(0, Height):
		
		for j in range(0, Width):
			
			#Four colors combination 
			
			if -22.5 <= Orientation[i, j] <= 22.5:
				
				OrtArray[i, j] = np.array([128, 128, 128])
				
				
			elif 22.5 < Orientation[i, j] <= 67.5:
				
				OrtArray[i, j] = np.array([0, 0, 155])
				
			elif 67.5 < Orientation[i, j] <= 90 or -90 <= Orientation[i, j] <= -67.5:
				
				OrtArray[i, j] = np.array([0, 155, 0])
				
			elif -67.5 < Orientation[i, j] <= -22.5:
				
				OrtArray[i, j] = np.array([155, 0, 0])
				
			else:
				
				OrtArray[i, j] = np.array([0, 0, 0])
				
				
	return (Magnitude, Orientation)




def BilinearInterpolation(image, x, y):
	
	#Round numbers down to the nearest integer of rows and columns 
	
	m = math.floor(x)
	
	n = math.floor(y)

	m1 = math.ceil(x)
	
	n1 = math.ceil(y)
	
	a = x - m
	
	b = y - n
	
	#image is still seen as a 2D array 
	
	Temp1 = (1 - a) * (1 - b) * image[m, n]
	
	
	if m1 < image.shape[0]:
		
		Temp2 = a * (1 - b) * image[m1, n]
		
	else:
		
		Temp2 = 0 * image[0, 0]
		
		
	if n1 < image.shape[1]:
		
		Temp3 = (1 - a) * b * image[m, n1]
		
	else:
		
		Temp3 = 0 * image[0, 0]
		
		
	if m1 < image.shape[0] and n1 < image.shape[1]:
		
		Temp4 = a * b * image[m1,n1]
		
	else:
		
		Temp4 = 0 * image[0, 0]
		
	return Temp1 + Temp2 + Temp3 + Temp4





#the implementation of the FindPeaksImage function



def FindPeaksImage(image, thres):
	
	
	image1 = Image.open(image)
	
	Y = image1.width
	
	X = image1.height
	
	#load image 
	
	Pixel = image1.load()
	
	#initialize the result image as empty 2D array
	
	ResultImage = np.zeros((X, Y), dtype=np.float32)
	
	#The edge magnitude and orientations can be computed using the Sobel filter you just implemented.
	
	
	Magnitude, Orientation= SobelImage(image)
	
	
	for i in range(X):
		
		for j in range(Y):
			
			if Magnitude[i, j] < thres:
				
				Magnitude[i, j] = 0
				
			else:
				
				Ort = Orientation[i, j]
				
				#A peak response is found by comparing a pixel’s edge magnitude to 
				#two samples per- pendicular to an edge at a distance of one pixel
				#in EdgeDetection), call these two samples e0 and e1. 
				#Compute e0 and e1 using BilinearInterpolation.
			
				
				e0 = []
			
				e1 = []
				
				e0.append(i + math.cos(math.radians(Ort)))
				
				e0.append(j + math.sin(math.radians(Ort)))
				
				
				e1.append(i + math.cos(math.radians(-Ort)))
				
				e1.append(j + math.sin(math.radians(-Ort)))
				
				#obtaining the magnitudes of these two samples 
				
				e0Magnitude = BilinearInterpolation(Magnitude, e0[0], e0[1])
				
				e1Magnitude = BilinearInterpolation(Magnitude, e1[0], e1[1])
				
				if e0Magnitude < Magnitude[i, j] > e1Magnitude:
					
					ResultImage[i, j] = 255
					
				else:
					
					ResultImage[i, j] = 0
					
	#same as imsave(new image) here 
					
	cv2.imwrite('7.png', ResultImage)

	
	
	
	
#Driver/Testing codes:
	
#Required Test:
	
#Find the peak responses in ”Circle.png” with thres = 40.0 and save as ”7.png”
	

a = FindPeaksImage("hw1_data/Circle.png",40.0)





	