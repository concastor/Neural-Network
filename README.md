# Neural-Network
Self coded Neural Network for personal projects


Written in python, it is a template for a three layered Neural network. the constructor takes in 3 numbers that decides the number of inputs, hidden nodes and outputs respectively. it uses the numpy library for the matrices and the matrix math.

# important functions:
  
  Guess_ff: this function will take in a list of numbers as the inputs and using a feed forward algorithm it will output a number between 0 and 1 for each output.
  
  train(currently not working): used for deep learning. it takes the error of the guess and uses back propagation to adjust the weight matrices by the learning rate
  
  Mutate: used for neuro-evolution. goes through each element in all the weight martices and mutates a percent of the elements, the percent is defined by the learning rate
  
  copy: also used for neuro-evolution, makes a deep copy of the current neural network and reuturns it to be used to create the next genereation
