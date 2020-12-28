# predict the bird name based on the output of the most recent model
import os
from functools import partial
from PIL import Image
import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model
from tensorflow.keras.applications import VGG16 as vgg16
from tensorflow.keras.applications.vgg16 import decode_predictions


# recent_model_path = 'saved_models'
# labels_converter = os.path.join('label_decoders', 'label_converter_initial_models.csv')

VGG16 = 'vgg16'
CUR_MODEL = 'cur_model'


def resnet101v2_decoder(probs):
    df = pd.read_csv('backend/ml/label_decoders/label_converter_initial_model.csv')
    print(df.keys())
    probs = list(enumerate(probs[0, :]))
    probs.sort(key=lambda x: x[1], reverse=True)
    probs = probs[:5]
    recovered_labels = list()
    for label_id, prob in probs:
        recovered_labels.append((df.loc[label_id]['0'], prob))
    print(f'We got probabilities {recovered_labels}')
    return [recovered_labels, ]

models = {
    VGG16: {
        'model': None,
        'init': partial(vgg16, input_shape=(224, 224, 3)),
        'decoder': partial(decode_predictions, top=5)
    },
    CUR_MODEL: {
        'model': None,
        'init': partial(load_model, 'backend/ml/saved_models'),
        'decoder': resnet101v2_decoder
    }
}

models_init_functions = {
    VGG16: partial(vgg16, input_shape=(224, 224, 3)),
    CUR_MODEL: partial(load_model, 'backend/ml/saved_models')
}


def load_model(model_name=CUR_MODEL):
    if models[model_name]['model'] is None:
        models[model_name]['model'] = models[model_name]['init']()

    return models[model_name]


def predict(model, path_to_img):
    img = Image.open(path_to_img)
    img = img.resize((224, 224))
    img = np.array(img)
    img = img / 255
    img = np.expand_dims(img, axis=0)
    preds = model['model'].predict(img)
    preds = model['decoder'](preds)
    return preds


if __name__ == '__main__':
    load_model(CUR_MODEL)
    print('Success')