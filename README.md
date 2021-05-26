# Pokemon detection
The programm uses machine learning (tensorflow in python) to take any image, scale it and return a prediced pokemon type. 

The main file to run is loadAndUseModel. Sadly my model object file was to large to upload here, so you will have to train the model yourself, locally.
You should be able to do this by running the MLpokemon.py file, at which point a model will be created (consisting of a model.index and model.data file), ready for use in the loadAndUseModel.py file.

Description of files:
loadAndUseModel.py - run this code with a relevant image called usedImg.png to try the model
MLpokemon.py - run this code to create a model. You can determine the number of epochs (for how long the model learns) at the bottom of the file
processedImg.png - automatically generated by the code, represents the image after altering dimensions
usedImg.png - The image to be scanned where the programm will determine what type of pokemon is in the image.
