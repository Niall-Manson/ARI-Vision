#non cnn related imports
import numpy as np
import polars as pl
import gc
import time

#disabling information (I) and warnings(W) for tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

#tf/keras (ml)
import tensorflow as tf
from keras.optimizers import Adam
from keras.callbacks import History
from sklearn.model_selection import train_test_split

test_df = pl.read_csv("C:/Users/Callum/OneDrive/Desktop/VS Code Projects/Python/ARK/AI/csv files/augmented_rgb_trashnet_128_divided/test11.csv")

test_data = np.array(test_df, dtype="float32")
test_data = np.transpose(test_data)

x_test = test_data[:, 1:] / 255 #pixel values - test
y_test = test_data[:, 0] #label - test

del test_df
del test_data

x_test = x_test.reshape(x_test.shape[0], *(128, 128, 4))

base_model = tf.keras.applications.xception.Xception(weights="imagenet", 
                                            include_top=False)


avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
output = tf.keras.layers.Dense(6, activation="softmax")(avg)


cnn_model = tf.keras.Model(inputs=base_model.input, outputs=output)

#compiling the cnn model
cnn_model.compile(
    loss = "sparse_categorical_crossentropy",
    optimizer = Adam(learning_rate=0.001),
    metrics = ["accuracy"] #maximising accuracy
)

with open("data.csv", "w") as f:
    f.write("acc,loss,val_acc,val_loss,test_acc,test_loss\n")


for j in range(10):
    for i in range(11):
        #splitting into train and test
        train_df = pl.read_csv(f"C:/Users/Callum/OneDrive/Desktop/VS Code Projects/Python/ARK/AI/csv files/augmented_rgb_trashnet_128_divided/test{i}.csv")

        #split the training and testing data into X (image) and Y (label) arrays
        #for x_train and y_train
        train_data = np.array(train_df, dtype="float32")
        train_data = np.transpose(train_data)

        x_train = train_data[:, 1:] / 255 #pixel values - train
        y_train = train_data[:, 0] #label - train

        # split the training data into train and validate arrays (will be used later)
        x_train, x_validate, y_train, y_validate = train_test_split(
            x_train, y_train, test_size=0.2, random_state=12345)

        del train_df
        del train_data

        x_train = x_train.reshape(x_train.shape[0], 128, 128, 4)
        x_validate = x_validate.reshape(x_validate.shape[0], *(128, 128, 4))

        #for noting acc and loss vals
        history = History()
        print("saved")

        #training the cnn model
        cnn_model.fit(
            x_train, y_train, batch_size=128, 
            epochs=1, 
            verbose=0,
            validation_data=(x_validate, y_validate), #how the ai tests itself
            callbacks=[history] #saving
        )
        
        test_loss, test_accuracy = cnn_model.evaluate(x_test, y_test, batch_size=128)
        
        data = f'{history.history["accuracy"][0]},{history.history["loss"][0]},{history.history["val_accuracy"][0]},{history.history["val_loss"][0]},{test_accuracy},{test_loss}'
        print(f"v4.{j}{i}")
        print(f'train acc:  {history.history["accuracy"][0]}')
        print(f'train loss: {history.history["loss"][0]}')
        print(f'val acc:    {history.history["val_accuracy"][0]}')
        print(f'val loss:   {history.history["val_loss"][0]}')
        print(f'test acc:   {test_accuracy}')
        print(f'test loss:  {test_loss}')
        print()

        with open("data.csv", "a") as f:
            f.write(f"{data}\n")
        
        print(f"{j},{i}")
        print()
        cnn_model.save(f"v4.{j}.{i}")
        
        gc.collect()
        time.sleep(30)