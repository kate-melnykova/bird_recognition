from numpy import round

from backend.ml.predict import predict, load_model, VGG16, CUR_MODEL


def recognize_bird(path_to_file):
    """
    1. Get image using the path to file and predict
    2. Decode prediction
    """
    model = load_model(CUR_MODEL)
    predictions = predict(model, path_to_file)[0]
    data = list()
    for prediction in predictions:
        print(f'prediction {prediction}')
        label = prediction[0]
        prob = prediction[1]
        prob = str(round(prob, 1))
        data.append((label, prob))
    return data
