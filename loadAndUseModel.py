import tensorflow as tf
import imageio
import numpy as np

def retriveModel():
    
    model = getModel()
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    model.load_weights('./model')
    
    return model

def getModel():
    from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
    model = tf.keras.Sequential()
    model.add(Conv2D(100, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(96, 96, 4)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(200, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(len(types), activation='softmax'))
    return model

def retriveDataSet():
    image = imageio.imread('processedImg.png')
    
    pokemon = np.array([np.array(image)])
    
    return pokemon

def resizeImg():
    from PIL import Image


    img = Image.open('usedImg.png')
    img = img.resize((96, 96), Image.ANTIALIAS) #basewidth, hsize
    img.save('processedImg.png')

def useModel():
    resizeImg()
    
    global types
    types = ['Normal', 'Grass', 'Fire', 'Water', 'Poison', 'Flying', 'Dragon', 'Bug', 'Electric', 'Ground', 'Fairy', 'Psychic', 'Fighting', 'Steel', 'Ice', 'Rock', 'Ghost', 'Dark']

    image = retriveDataSet()
    model = retriveModel()
    

    out = list(model.predict(image)[0])
    for i, element in enumerate(out):
        print(str(round(element*100, 2)) + '% for ' + types[i])
    
if __name__ == '__main__':
    useModel()