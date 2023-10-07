"""
20:49 05/05/23 - 0%

"""

#startup
print(""" 
 █████╗	   ██████╗    ██╗  ██╗   
██╔══██╗   ██╔══██╗   ██║ ██╔╝   
███████║   ██████╔╝   █████╔╝	
██╔══██║   ██╔══██╗   ██╔═██╗	
██║  ██║██╗██║  ██║██╗██║  ██╗██╗
╚═╝  ╚═╝╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═╝╚═╝
""")

print("\nLoading...")
#non ai related imports
import random
import time
import ast
import getpass
import hashlib

#non cnn related imports
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import cv2

#disabling information (I) and warnings(W) for tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

#tf/keras (ml)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split 

#main script
def main():
	print("Libraries successfully imported.")
	lines = update()
	enterPressed = False

	#password
	salt = lines[7]
	pass_hash = lines[6]
	
	
	# Comparing the attempt
	attempt = getpass.getpass("Enter Password >>")
	attempt_password = attempt+salt
	hashedattempt = hashlib.md5(attempt_password.encode())
	while pass_hash != hashedattempt.hexdigest():
		attempt = getpass.getpass("Incorrect Password. Please Try Again. >>")
		attempt_password = attempt+salt
		hashedattempt = hashlib.md5(attempt_password.encode())
	print("Access Granted.")

	print("\nFor help on how to use A.R.K. OS, type \"help\".")
	while True:
		#formatting the string to convert to list of length 3 and remove whitespace
		if enterPressed:
			strCommand = input(">>").strip()
			enterPressed = False
		else:
			strCommand = input("\n>>").strip()

		command = []
		commandLine = ""
		quotes = False
		isList = 0
		for i, letter in enumerate(strCommand):
			commandLine += letter
			if letter == " " and quotes == False and isList == 0:
				command.append(commandLine.strip())
				commandLine = ""
				
			if letter == "\"" and strCommand[i-1] != "\\":
				quotes = not(quotes)
			elif letter == "[" and quotes == False:
				isList += 1
			elif letter == "]" and quotes == False:
				isList -= 1
			
		command.append(commandLine)
		while len(command) < 3:
			command.append("")

		#the commands	  
		#update and download
		if command[0] in ["update", "download", "view", "changePassword", "exit"]:
			if command[1] == "":
				if command[0] == "update":
					lines = update()
					
				elif command[0] == "download":
					try:
						x_train, y_train, x_validate, y_validate = download(lines[2])
					except FileNotFoundError:
						print(f"The file path {lines[2]} was not found.")
						
				elif command[0] == "view":
					view(lines)
					
				elif command[0] == "changePassword":
					changePassword(lines)
					
				else:
					exit()
				
			else:
				print(f"Did not expect \"{command[1]}\" after \"{command[0]}\".")
		
		#ai
		elif command[0] == "ai":
			if command[1] in ["create", "train", "info", "load", "evaluate", "predict"]:
				if command[2] == "":
					if command[1] == "create":
						try:
							cnn_model, cnn_structure, learning_rate = model_create(ast.literal_eval(lines[3]), lines[5])
							print(f"AI successfully created with a structure of {ast.literal_eval(lines[3])} and a learning rate of {float(lines[5])}.")
						except UnboundLocalError:
							print("The dataset has not been downloaded.")
					
					elif command[1] == "train":
						if "x_train" in locals():
							try:
								cnn_model = model_train(cnn_model, 
														x_train, y_train, x_validate, y_validate, 
														lines[0], lines[1], lines[4], lines[8][1:-1])
							except UnboundLocalError:
								print("The AI has not been created.")
						else:
							print("The dataset has not been downloaded.")
					
					elif command[1] == "load":
						cnn_model, cnn_structure, learning_rate = model_create(findStructure(lines[8]), 0.01)
						cnn_model.load_weights( lines[8][1:-1] + "/cp.ckpt")
						print("AI successfully loaded.")
					
					elif command[1] == "evaluate":
						if "x_train" in locals():
							try:
								model_evaluate(cnn_model, x_train, y_train, lines[1])
							except UnboundLocalError:
								print("The AI has not been created.")
						else:
							print("The dataset has not been downloaded.")

					elif command[1] == "predict":
						model_predict(cnn_model)

					else:
						cnn_model.summary()
						
				else:   
					print(f"Did not expect \"{command[2]}\" after \"start\".")
			else:
				print(f"Did not expect \"{command[1]}\" after \"ai\".")
		
		#length check (assuming its not help, download, update or ai
		elif len(command) > 3:
			print(f"Did not expect \"{command[3]}\" after \"{command[2]}\".")
		
		#vars from config.txt
		elif command[0] in ["epochs", "batch_size", "learning_rate"]:
			lines = setAndChange(lines, command, "epochs", 0)
			lines = setAndChange(lines, command, "batch_size", 1)
			lines = setAndChange(lines, command, "learning_rate", 5)
		
		elif command[0] == "verbose":
			if command[1] == "set":
				if command[2] in ["0", "1", "2"]:
					lines[4] = command[2]
					file = open("config.txt", "w")
					file.write("\n".join(lines))
					file.close()
					
					print(f"verbose = {lines[4]}")
				else:
					print("Expected values 0, 1 or 2 after \"set\"")
			else:
				print(f"Did not expect \"{command[1]}\" after \"verbose\".")
		
		elif command[0] in ["path_to_csv", "cnn_name"]:
			if command[1] == "set":
				if "\"" not in command[2]:
					command[2] = f"\"{command[2]}\""
				
				if command[0] == "path_to_csv":
					lines = stringVar(lines, command, 2)
					print(f"path_to_csv = {lines[2]}")
				else:
					lines = stringVar(lines, command, 8)
					print(f"cnn_name = {lines[8]}")

			else:
				print("Expected \"set\" after \"path_to_csv\".")
		
		elif command[0] == "cnn_structure":
			if command[1] == "set":
				try: 
					lst = ast.literal_eval(str(command[2]))
					notInts = []
					for item in lst:
						if not(isinstance(item, int)):
							notInts.append(item)
					if len(notInts) == 0:
						lines = stringVar(lines, command, 3)
						print(f"cnn_structure = {lines[3]}")
						
					else:
						if len(notInts) == 1:
							print(f"Expected item \"{notInts[0]}\" in {lst} to be of data type integer.")
						else:
							print("Expected items \"" + "\", \"".join(notInts[:-1]) + "\" and \"" + notInts[-1] + "\" to be of data type integer.")
				except (ValueError, SyntaxError):
					print("Expected value with data type list after \"set\".") 
			else:
				print("Expected \"set\" after \"cnn_structure\".")
		
		elif command[0] == "":
			enterPressed = True

		else:
			print(f"\"{command[0]}\" is not a known command.")

#updating vars from config
def update():
	file = open("config.txt", "r")
	lines = []
	for line in file:
		lines.append(line.replace("\n", ""))
	
	print("Variables successfully updated from config.txt.")
	return lines

#viewing all changeable variables		
def view(lines):
	print(f"epochs = {lines[0]}\n" + 
		  f"batch_size = {lines[1]}\n" + 
		  f"path_to_csv = {lines[2]}\n" + 
		  f"cnn_structure = {lines[3]}\n" + 
		  f"verbose = {lines[4]}\n" +
		  f"learning_rate = {lines[5]}\n" + 
	      f"cnn_name = {lines[8]}") 
		  
#changing the password
def changePassword(lines):
	salt = lines[7]
	pass_hash = lines[6]
	attempt = getpass.getpass("Enter Password >>")
	attempt_password = attempt+salt
	hashedattempt = hashlib.md5(attempt_password.encode())
	while pass_hash != hashedattempt.hexdigest():
		if attempt != "":
			attempt = getpass.getpass("Incorrect Password. Please Try Again. >>")
			attempt_password = attempt+salt
			hashedattempt = hashlib.md5(attempt_password.encode())
		else:
			print("Command ended.")
			return
		
	newpass = getpass.getpass("New Password >>")
	newpass_password = newpass+salt
	hashednewpass = hashlib.md5(newpass_password.encode())
	
	newpass2 = getpass.getpass("Re-enter New Password >>")
	newpass2_password = newpass+salt
	hashednewpass3 = hashlib.md5(newpass2_password.encode())
	while newpass != newpass2:
		if newpass2 != "":
			newpass2 = getpass.getpass("Do Not Match. Re-enter New Password >>")
			newpass2_password = newpass+salt
			hashednewpass2 = hashlib.md5(newpass2_password.encode())
		else:
			print("Command ended.")
			return
	
	lines[7] = hashednewpass2.hexdigest()
	file = open("config.txt", "w")
	file.write("\n".join(lines))
	return lines

#var setting and changing
def setAndChange(lines, command, varName, slice):
	if command[0] == varName:
		if command[1] == "set" or command[1] == "change":
			if command[2].lstrip("+-").isdigit():
				lines = intVar(lines, int(lines[slice]), command, slice)
				print(f"{varName} = {int(lines[slice])}")
				
			else:
				print(f"Expected integer after \"{varName} {command[1]}\".")
		else: 
			print(f"Expected \"set\" or \"change\" after \"{varName}\".")
	
	return lines

#changing/setting integer variables
def intVar(lines, var, command, slice):
	if command[1] == "set":
		var = int(command[2])
	else:
		var += int(command[2])
	
	lines[slice] = str(var)
	file = open("config.txt", "w")
	file.write("\n".join(lines))

	return lines

#setting string/list variables
def stringVar(lines, command, slice):
	lines[slice] = command[2]
	file = open("config.txt", "w")
	file.write("\n".join(lines))
	return lines

#finding cnn_structure from cnn_name
def findStructure(cnn_name):
	start = cnn_name.index("[")
	end = cnn_name.index("]")+1
	cnn_structure = ast.literal_eval(cnn_name[start:end])
	return cnn_structure

#reading csv file and formatting arrays
def download(path_to_csv):
	#downloading test files
	train_df = pl.read_csv(path_to_csv[1:-1])
	
	#split the training and testing data into X (image) and Y (label) arrays
	#for x_train and y_train
	train_data = np.array(train_df, dtype="float32")
	train_data = np.transpose(train_data) #flipping the rows and columns for polars
	
	x_train = train_data[:, 1:] / 255 #pixel values
	y_train = train_data[:, 0] #label

	# split the training data into train and validate arrays (will be used later)
	x_train, x_validate, y_train, y_validate = train_test_split(
		x_train, y_train, test_size=0.2, random_state=12345)

	x_train = x_train.reshape(x_train.shape[0], *(128, 128, 1))
	print(x_train.shape)

	x_validate = x_validate.reshape(x_validate.shape[0], *(128, 128, 1))
	
	print(f"File from csv file {path_to_csv} successfully downloaded.")
	return x_train, y_train, x_validate, y_validate

#defining and compiling a cnn_model
def model_create(cnn_structure, learning_rate):
	#defining the model
	cnn_model = Sequential([
		Conv2D(filters=32, kernel_size=3, activation="relu", input_shape=(128, 128, 1)),
		MaxPooling2D(pool_size=2),
		Dropout(0.2),
		Flatten(),
	])
	
	#adding dense layers
	for nodes in cnn_structure:
		cnn_model.add(Dense(nodes, activation = "relu"))
	cnn_model.add(Dense(28, activation = "softmax"))
	
	#compiling the cnn model
	cnn_model.compile(
		loss = "sparse_categorical_crossentropy",
		optimizer = Adam(learning_rate=float(learning_rate)),
		metrics = ["accuracy"] #maximising accuracy
	)
	
	return cnn_model, cnn_structure, learning_rate

#training the AI
def model_train(cnn_model, x_train, y_train, x_validate, y_validate, epochs, batch_size, verbose, cnn_name):
	#saving the model setup
	checkpoint_path = cnn_name + "/cp.ckpt"
	checkpoint_dir = os.path.dirname(checkpoint_path)
	cp_callback = ModelCheckpoint(filepath=checkpoint_path,
		save_weights_only=True,
		verbose=1
	)
	
	#training the cnn model
	cnn_model.fit(
		x_train, y_train, batch_size = int(batch_size), #higher batch size -> lower run time, more processing power from gpu
		epochs = int(epochs), #worse gpu -> more epochs to compensate for less convolutional layers/neurons
		verbose = int(verbose), #0 -> nothing printed
								#1 -> epoch num, loss & accuracy values, loading bar
								#2 -> epoch num, loss & accuracy values
		validation_data = (x_validate, y_validate), #how the ai tests itself
		callbacks=[cp_callback] #saving
	)
	
	print(f"AI successfully trained with {int(epochs)} epochs and a batch size of {int(batch_size)}.")
	return cnn_model

#evaluating the accuracy of the model
def model_evaluate(cnn_model, x_train, y_train, batch_size):
	loss, accuracy = cnn_model.evaluate(x_train, y_train, batch_size=int(batch_size))
	print(f"This model can identify the correct output with {round(accuracy, 2)*100}% accuracy.")

#predicting the answer to an image
def model_predict(cnn_model):
	#capturing the image
	inp = input()
	image = cv2.imread(inp)
	
	#cropping to a 128x128 file size
	height = image.shape[0]
	width = image.shape[1]

	if height < width:
		image = image[:, round((width-height)/2):round((width-height)/2+height)]
	elif height > width:
		image = image[round((height-width)/2):round((height-width)/2+width), :]
	image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)

	#obtaining the pixels in greyscale
	pixelValues = np.array([], dtype="float32")
	for i in range(128):
		for j in range(128):
			rgbPixel = image[i, j] 
			greyscalePixel = sum(rgbPixel)/3/255
			pixelValues = np.append(pixelValues, [greyscalePixel])
	
	#formatting the image to an array of shape (1, 128, 128, 1)
	formattedPixels = np.array([pixelValues])
	formattedImage = pixelValues.reshape(*(1, 128, 128, 1))
	
	#predicting the material of the object
	prediction = cnn_model.predict(formattedImage)
	
	#using the prediction to give an output
	labels = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
	label = labels[np.argmax(prediction)]
	print(prediction)
	print(f"Predicted {label} with {round(np.amax(prediction)*100, 2)}% certainty.")
	
	#saving the image taken for debugging
	cv2.imwrite("test0.png", image)

main()