import os
from joblib import load

def load_num_model():
    path = os.path.join(os.getcwd(), "saved_model_params", "lr_model.joblib")
    return load(path)