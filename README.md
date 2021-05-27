# Pokemon detection
The programm uses machine learning (tensorflow in python) to take any image, scale it and return a prediced pokemon type. 

How to use:
0. Make sure you have installed the python libraries imageio, tensorflow and numpy.
1. Extract the zip file PokeSprites.zip
2. Inside the new folder called PokeSprites, you will find another folder called PokeSprites. Copy this second folder into the directory of the other files. 
3. Run the MLpokemon.py file to create a model. For how long you want to train can be adjustet by changing the epochs variable at the bottom of the programm (default is 100, more should not be nessecary)
4. Place your image in the folder and call it usedImg.png.
5. Run the loadAndUseModel.py file to use the model. 
6. Repeat 4-5 for any images you want to analyse (no need to make a new model)


Overview of files:

loadAndUseModel.py - run this code with a relevant image called usedImg.png to try the model

MLpokemon.py - run this code to create a model. You can determine the number of epochs (for how long the model learns) at the bottom of the file

PokeSprites.zip - A zip folder of pokemon sprites to be used for training the model

pokemon.csv - A file of what type each pokemon is
