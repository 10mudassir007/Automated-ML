import pickle

with open(r"F:\Files\Portfolio\AUTOML\models\median_house_value_model.pkl", "rb") as f:
    model = pickle.load(f)

with open(r"F:\Files\Portfolio\AUTOML\models\median_house_value_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

