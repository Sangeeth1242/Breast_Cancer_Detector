import pandas as pd
import numpy as nm
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


def detect(mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness):
    data = pd.read_csv('Breast_cancer_data.csv')
    x = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    train_x, test_x, train_y, test_y = train_test_split(
        x, y, random_state=0, test_size=0.2)
    model = KNeighborsClassifier()
    model.fit(train_x, train_y)
    y_detection = model.predict(
        [[mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness]])
    print('diagnosis : ', y_detection)
    y_test_detection = model.predict(test_x)
    print('accuracy : ', r2_score(test_y, y_test_detection))


mean_radius = float(input('radius : '))
mean_texture = float(input('texture : '))
mean_perimeter = float(input('perimeter : '))
mean_area = float(input('area : '))
mean_smoothness = float(input('smoothness : '))
detect(mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness)
