This neural network written in python uses the VGG16 tensorflow/keras algorithm to identify pictures of cats vs dogs. 

How to initialize:
Go to https://www.kaggle.com/c/dogs-vs-cats and download the dataset. 
Download either the py or ipynb file and run. Save in the same directory.
Change the paths in the file to the directory above.
Run code.

How to run on new images:
Load, compile and train the model. Then run the code "predictions = model.predict(reshaped, verbose=0)". An output of 0 indicate cat and an output of 1 indicates dog.