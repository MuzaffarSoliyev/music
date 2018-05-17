from keras.models import load_model
from keras.utils import plot_model
from IPython.display import SVG
from keras.utils.vis_utils import plot_model
from keras.utils.vis_utils import model_to_dot


model = load_model("../models/muzaffar.h5")
plot_model(model, to_file='model_plot1.png', show_shapes=True, show_layer_names=True)
SVG(model_to_dot(model).create(prog='dot', format='svg'))
print(model_to_dot(model))
