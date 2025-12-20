import pickle

with open(r"F:\Files\Portfolio\AUTOML\models\median_house_value_model.pkl", "rb") as f:
    model = pickle.load(f)

with open(r"F:\Files\Portfolio\AUTOML\models\median_house_value_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

array = [[-114.310000,34.190000,15.000000,5612.000000,1283.000000,1015.000000,472.000000,1.493600]]
print()
print(model.predict(scaler.transform(array)))