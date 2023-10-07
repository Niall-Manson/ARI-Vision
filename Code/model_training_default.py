#non cnn related imports
import numpy as np
import polars as pl
import gc

#disabling information (I) and warnings(W) for tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

#tf/keras (ml)
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import History, LearningRateScheduler
from sklearn.model_selection import train_test_split

def main():
    print("Library set-up complete.")
    J_TRUE = 0
    I_TRUE = 0
    
    #change these depending on how you want to train the model
    TRAIN_FROM_SCRATCH = False
    LOAD_MODEL = False

    if TRAIN_FROM_SCRATCH:
        model = model_create()
        model = model_compile(model)
        data_create()

    elif LOAD_MODEL:
        model = load_model("C:/Users/Callum/OneDrive/Desktop/VS Code Projects/Python/ARK/AI/AI Models (Final)/v4/v4.11.6")
        J_TRUE = 11 #j = b where latest model is titled va.b.c
        I_TRUE = 6 #i = c where latest model is titled va.b.c
        I_TRUE += 1

    else:
        model = create_pretrained_model()
        model = model_compile(model)
        data_create()

    x_test, y_test = csv_download(11)
    x_test = x_reshape(x_test)

    print("Model set-up complete.")
    for j in range(100):
        for i in range(11):
            #getting to right version
            if j < J_TRUE or (j == J_TRUE and i < I_TRUE):
                continue

            x_train, y_train = csv_download(i)
            x_train, x_validate, y_train, y_validate = validate_creation(x_train, y_train)
            x_train = x_reshape(x_train)
            x_validate = x_reshape(x_validate)

            print(f"v4.{j}.{i} began.")
            model, train_acc, train_loss, val_acc, val_loss, lr = model_train(model, x_train, y_train, x_validate, y_validate)
            test_acc, test_loss = model_evaluate(model, x_test, y_test)
            
            data_append(train_acc, train_loss, val_acc, val_loss, test_acc, test_loss)
            model.save(f"v4.{j}.{i}")

            gc.collect()

            print(f"v4.{j}.{i} complete.")
            print(f'train acc:  {train_acc}')
            print(f'train loss: {train_loss}')
            print(f'val acc:    {val_acc}')
            print(f'val loss:   {val_loss}')
            print(f'test acc:   {test_acc}')
            print(f'test loss:  {test_loss}')
            print(f"lr:         {lr}")
            print()

def model_create():
    model = Sequential([
        Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(1, 128, 128, 4)),
        MaxPooling2D(pool_size=2),
        Dropout(0.2),
        Flatten(),
        
        Dense(120, activation='relu'),
        Dense(84, activation='relu'),
        Dense(6, activation='softmax')
    ])
    return model

def model_compile(model):
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(learning_rate=0.001),
        metrics=['accuracy']
    )
    return model

def create_pretrained_model():
    base_model = tf.keras.applications.xception.Xception(weights="imagenet",  # Load weights pre-trained on ImageNet.
        include_top=False
    )
    avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    output = tf.keras.layers.Dense(6, activation="softmax")(avg)

    model = tf.keras.Model(inputs=base_model.input, outputs=output)
    return model

def csv_download(testNo):
    df = pl.read_csv(f"C:/Users/Callum/OneDrive/Desktop/VS Code Projects/Python/ARK/AI/csv files/augmented_rgb_trashnet_128_divided/test{testNo}.csv")

    data = np.array(df, dtype='float32')
    data = np.transpose(data)
    
    x = data[:, 1:] / 255
    y = data[:, 0]

    return x, y

def validate_creation(x_train, y_train):
    x_train, x_validate, y_train, y_validate = train_test_split(
        x_train, y_train, test_size=0.2, random_state=12345,
    )
    return x_train, x_validate, y_train, y_validate

def x_reshape(x):
    x = x.reshape(x.shape[0], 128, 128, 3)
    return x

def lr_schedule(lr, epochs):
    if epochs > 75:
        return lr
    else:
        return lr - 0.0000666667 #y = 0.0000666667x - 0.006

def model_train(model, x_train, y_train, x_validate, y_validate):
    history = History()
    schedule = LearningRateScheduler(lr_schedule)
    model.fit(
        x_train, y_train, 
        batch_size=16, 
        epochs=1, 
        verbose=1,
        validation_data=(x_validate, y_validate), #how the ai tests itself
        callbacks=[history, #history: saving
                   schedule] #schedule: lr scheduling
    )
    acc = history.history["accuracy"][0]
    loss = history.history["loss"][0]
    val_acc = history.history["val_accuracy"][0]
    val_loss = history.history["val_loss"][0]
    lr = round(model.optimizer.lr.numpy(), 6)
    return model, acc, loss, val_acc, val_loss, lr 

def model_evaluate(model, x_test, y_test):
    test_loss, test_acc = model.evaluate(x_test, y_test, 
        batch_size=16, 
        verbose=1
    )
    return test_acc, test_loss

def data_create():
    with open("data.csv", "w") as f:
        f.write("acc,loss,val_acc,val_loss,test_acc,test_loss\n")

def data_append(train_acc, train_loss, val_acc, val_loss, test_acc, test_loss):
    data = f"{train_acc},{train_loss},{val_acc},{val_loss},{test_acc},{test_loss}"
    with open("data.csv", "a") as f:
            f.write(f"{data}\n")

main()