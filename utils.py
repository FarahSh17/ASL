import pickle
from keras.models import load_model

# Load a variable from a file using pickle
def load_variable(filepath):
    with open(filepath, 'rb') as file:
        variable = pickle.load(file)
    return variable


# Load a model 
def load_aslmodel(filepath):
    model = load_model(filepath)
    return model