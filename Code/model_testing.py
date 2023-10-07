#non tf imports
import numpy as np
import polars as pl
import cv2
from rembg import remove
from skimage.color import rgba2rgb 

#disabling information (I) and warnings(W) for tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

#tf and similar imports
import tensorflow as tf

print("Imports loaded")

def main():
    cam = cv2.VideoCapture(0)
    model = model_load("C:/Users/Callum/OneDrive/Desktop/VS Code Projects/Python/ARK/AI/AI Models (Final)/v4/v4.11.6")
    print("model loaded")
    
    print(model_test(model, "c:/Users/Callum/OneDrive/Desktop/VS Code Projects/Python/ARK/AI/csv files/augmented_rgb_trashnet_128_divided/test11s.csv"))
    
    return
    
    while True:    
        try:
            image = preprocessing(input(), cam)
            prediction = model_predict(model, image)
        except AttributeError:
            print("invalid input")

def model_load(filepath):
    return tf.keras.models.load_model(filepath)

def model_test(model, filepath):
    test_df = pl.read_csv(filepath)
	
    test_data = np.array(test_df, dtype="float32")
    test_data = np.transpose(test_data) #flipping the rows and columns for polars
	
    x_test = test_data[:, 1:] / 255 #pixel values
    y_test = test_data[:, 0] #label   
        
    x_test = x_test.reshape(x_test.shape[0], *(128, 128, 3))
    _, test_accuracy = model.evaluate(x_test, y_test, batch_size=128, verbose=0)
    return test_accuracy

def take_image(cam):
    cv2.namedWindow("test")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("test", frame)

        k = cv2.waitKey(1)
        if k%256 == 32:
            # SPACE pressed
            break

    cv2.destroyAllWindows()    
    return frame

def remove_bg(image):
    image = remove(image)
    image = rgba2rgb(image)

    image = np.where(image == (255, 255, 255), 0, image)

    return image

def image_show_large(image):
	temp_image = cv2.resize(image, (512, 512), interpolation = cv2.INTER_AREA)
	cv2.imshow("test", temp_image)
	cv2.waitKey(0)

def preprocessing(filepath, cam):
    if filepath == "":
        image = take_image(cam)
    else:
        if filepath[0] == '"':
            filepath = filepath[1:-1]
        image = cv2.imread(filepath, 1)

    height = image.shape[0]
    width = image.shape[1]

    if height < width:
        image = image[:, round((width-height)/2):round((width-height)/2+height)]
    elif height > width:
        image = image[round((height-width)/2):round((height-width)/2+width), :]
    image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
    image = remove_bg(image)
    image_show_large(image)

    pixelValues = np.array([], dtype="float32")
    for h in range(128):
        for w in range(128):
            for c in range(3):
                pixel = image[h][w][c]/255
                pixelValues = np.append(pixelValues, [pixel])
    
    formattedImage = pixelValues.reshape(*(1, 128, 128, 3))

    return formattedImage

def model_predict(model, image):
    prediction = model.predict(image)

    labels = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
    label = labels[np.argmax(prediction)]
    print(prediction)
    print(f"Predicted {label} with {round(np.amax(prediction)*100, 2)}% certainty.")

    return prediction

main()