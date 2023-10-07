#importing - needed libraries
import cv2
import os

#important
import numpy as np
from skimage.util import random_noise
from skimage.color import rgba2rgb
import random
from rembg import remove

#nice to have
import time
import datetime

def main():
	csv_creating(datasetFilePath="C:/Users/Callum/OneDrive/Desktop/VS Code Projects/Python/ARK/AI/dataset-resized", 
		  		 csvFilePath="C:/Users/Callum/OneDrive/Desktop/VS Code Projects/Python/ARK/AI/csv files/augmented_rgb_trashnet_128_divided/",
				 pixelColumns=4, dataAugmentation=16, size=128, fileNo=12, removeBg=True)

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
	
	# apply automatic Canny edge detection using the computed median
	lower = int( max(0, (1.0 - sigma) * v))
	upper = int( min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	
	# return the edged image
	return edged

def image_show_large(image):
	temp_image = cv2.resize(image, (512, 512), interpolation = cv2.INTER_AREA)
	cv2.imshow("image_show_large", temp_image)
	cv2.waitKey(0)

def remove_white_bg(image):
	binaryImage = rembg_white(image)
	filledBinaryImage = filling_holes_fast(binaryImage)
	finalImage = applying_binaryImage(image, filledBinaryImage)

	return finalImage

#outputs estimation of background in terms of a binary image, without the obviously incorrect holes filled
def rembg_white(image):
	rembgImage = remove(image)

	binaryImage = image
	binaryImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	height = image.shape[0]
	width = image.shape[1]

	for y in range(height):
		for x in range(width):
			#pseudocode:
			#if image pixel is greyscale (i.e. range between rgb vals is less than 30) and rembg believes the pixel to be background: 0, else: 255
			binaryImage = np.where(max(image[y][x])-min(image[y][x]) < 30 and sum(rembgImage[y][x]) == 0, 0, 255)
			
	return binaryImage

#fills holes, returns updated binary image
def filling_holes_fast(binaryImage):
	height = binaryImage.shape[0]
	width = binaryImage.shape[1]
	
	for y in range(height):
		for x in range(width):
			if binaryImage[y][x] == 0:
				mask = np.zeros((height+2, width+2), np.uint8)
				cv2.floodFill(binaryImage, mask, (x,y), 100) #temp, removed before moving onto next pixel

				firstRow = binaryImage[0]
				lastRow = binaryImage[-1]
				firstColumn = binaryImage[:, 0]
				lastColumn = binaryImage[:, -1]
				edges = np.concatenate((firstRow, lastRow, firstColumn, lastColumn), axis=None)
				
				#replaces where pixel=100 with correct val (50 when background, 255 when foreground)
				if 100 in edges:
					binaryImage = np.where(binaryImage == 100, 50, binaryImage) #will be (0, 0, 0, 0) in finalImage
				else:
					binaryImage = np.where(binaryImage == 100, 255, binaryImage)

	return binaryImage

#uses binary image to know where to remove background in originals
def applying_binaryImage(image, filledBinaryImage):
	image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
	height = image.shape[0]
	width = image.shape[1]

	for y in range(height):
		for x in range(width):
			if filledBinaryImage[y][x] == 50:
				image[y][x] = (0, 0, 0, 0) #3rd index is alpha (transparency)

	return image

def csv_creating(datasetFilePath, csvFilePath, pixelColumns, dataAugmentation, size, fileNo, removeBg):
	#headers
	headers = ["label"]
	if pixelColumns == 1:
		for i in range(size**2):
			headers.append(f"pixel{i}")
	else:
		ALPHABET = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
		for i in range(size**2):
			for j in range(pixelColumns):
				headers.append(f"pixel{i}{ALPHABET[j]}")

	for i in range(fileNo):
		with open(f"{csvFilePath}test{i}.csv", "w") as file:
			file.write(",".join(headers)+"\n")

	folderList = os.listdir(datasetFilePath)

	#finding no of files
	totalFiles = 0
	completedFiles = 0
	for folderName in folderList:
		totalFiles += len(os.listdir(f"{datasetFilePath}/{folderName}"))

	start = time.time() #for loading bar

	for i, folderName in enumerate(folderList):
		fileList = os.listdir(f"{datasetFilePath}/{folderName}")
		
		for m, fileName in enumerate(fileList):
			#opening the image in greyscale
			if pixelColumns == 0:
				image = cv2.imread(f"{datasetFilePath}/{folderName}/{fileName}", 0)
			else:
				image = cv2.imread(f"{datasetFilePath}/{folderName}/{fileName}", 1)
			
			if removeBg:
				image = remove_white_bg(image)

			for n in range(dataAugmentation):
				#data augmentation
				if n == 0: #no change
					pass 
				elif n == 1: #vertical flip
					image = cv2.flip(image, 0) 
				elif n == 2: #horizontal flip
					image = cv2.flip(image, -1)
				elif n == 3: #horizontal and vertical flip
					image = cv2.flip(image, 0)
				elif n == 4: #90 degree rotation
					image = cv2.flip(image, -1)
					image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
				elif n == 5: #270 degree rotation
					image = cv2.rotate(image, cv2.ROTATE_180)
				elif n == 6: #adding "salt and pepper" noise (0.01)
					image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
				
				if n >= 6 and n <= 15 and n % 2 == 0: #adding "salt and pepper" noise (0.02)
					image = random_noise(image, mode='s&p',amount=0.02)

				if n >= 7 and n <= 15: #180 degree rotation
					image = cv2.rotate(image, cv2.ROTATE_180)

				#cropping to (size, size)
				height = image.shape[0]
				width = image.shape[1]

				if height < width:
					image = image[0:height, round((width-height)/2):round((width-height)/2+height)]
				elif height > width:
					image = image[round((height-width)/2):round((height-width)/2+width), 0:width]

				image = cv2.resize(image, (size, size), interpolation = cv2.INTER_AREA)

				#coverting the image from greyscale to individual pixel values
				if pixelColumns == 0:
					pixelValues = [str(i)]
					for j in range (image.shape[0]):
						for k in range (image.shape[1]):
							if str(image[j][k]).isdigit():
								pixelValues.append(str(image[j][k]))
							else: 
								pixelValues.append(str(int(image[j][k]*255)))
				else:
					pixelValues = [str(i)]
					for j in range (image.shape[0]):
						for k in range (image.shape[1]):
							for l in range(pixelColumns):
								if str(image[j][k][l]).isdigit():
									pixelValues.append(str(image[j][k][l]))
								else: 
									pixelValues.append(str(int(image[j][k][l]*255)))
	
				with open(f"{csvFilePath}test{random.randint(0, fileNo-1)}.csv", "a") as file:
					file.write(",".join(pixelValues)+"\n")

			completedFiles += 1
			if (m+1) % round(75/dataAugmentation) == 0:
				timeTaken = time.time()-start
				print(f"{i}: {m+1}/{len(fileList)}")
				print(f"{round((completedFiles)/totalFiles*100, 2)}%")
				print(f"Time elapsed: {datetime.timedelta(seconds=round(timeTaken, 2))}s/{datetime.timedelta(seconds=round(timeTaken/((completedFiles)/totalFiles), 2))}s")
				print()

			elif m == len(fileList)-1:
				print(f"{i}: {m+1}/{m+1}")

main()