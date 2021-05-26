import imageio
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, BatchNormalization, Activation

def retriveDataSet():
    global types
    types = ['Normal', 'Grass', 'Fire', 'Water', 'Poison', 'Flying', 'Dragon', 'Bug', 'Electric', 'Ground', 'Fairy', 'Psychic', 'Fighting', 'Steel', 'Ice', 'Rock', 'Ghost', 'Dark']
    pokemon = []
    
    for i in range(649):
        i += 1
        image = imageio.imread('./PokeSprites/'+str(i)+'.png')
        pokemon.append(np.array(image))
        
    with open('pokemon.csv', mode='r') as file:
        data = file.read()
        data = data.split('\n')
        data = [d.split(',') for d in data]
        data = [d for d in data if d != ['']]
        data = [[d[0], d[2]] for d in data]
    
    del data[0] #First row lables the data
    
    registeredNumbers = []
    labelData = []
    for d in data:
        if not d[0] in registeredNumbers:
            registeredNumbers.append(d[0])
            labelData.append(d)    
            
    forbiddenIndexes = [i for i, d in enumerate(labelData) if d[1] == 'Dragon']
    
    labelData = [types.index(d[1]) for i, d in enumerate(labelData) if not i in forbiddenIndexes]
    
    pokemon = [p for i, p in enumerate(pokemon) if not i in forbiddenIndexes]
    
    labelData = np.array(labelData)
    pokemon = np.array(pokemon)

    
    return pokemon, labelData[:len(pokemon)]

def main(epochs):
    x_train, y_train = retriveDataSet()

    x_train = x_train.reshape(x_train.shape[0], 96, 96, 4)
    input_shape = (96, 96, 4)

    model = tf.keras.Sequential()
    model.add(Conv2D(100, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(96, 96, 4)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(200, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(len(types), activation='softmax'))
    
    opt = tf.keras.optimizers.SGD(lr=0.02, momentum=0.9)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=epochs)
    
    
    
    loss, acc = model.evaluate(x_train, y_train, verbose=2)
    print(loss, acc)
    
    model.save_weights('./model')

if __name__ == '__main__':
    epochs = 100
    main(epochs)