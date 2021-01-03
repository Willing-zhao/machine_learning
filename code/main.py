import sys
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from PyQt5 import QtWidgets
from visual_interface import UI


def load_cnn_model():

    from model import CNN3
    model = CNN3()
    model.load_weights('../models/cnn3_best_weights.h5')
    return model


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    form = QtWidgets.QMainWindow()
    model = load_cnn_model()
    ui = UI(form, model)
    form.show()
    sys.exit(app.exec_())
