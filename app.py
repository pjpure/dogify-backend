# import the necessary packages

from PIL import Image
import numpy as np
import flask
import io
from keras.models import load_model
from flask_cors import CORS, cross_origin


# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
model = load_model('dogify_v13_in3_9533.h5')
width = 299


@app.route('/')
def hello():
    return "Dogify v9"


def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # # resize the input image and preprocess it
    image = image.resize(target)
    image = np.array(image)
    image = image.astype('float32')
    image /= 255
    image = np.reshape(image, (1, width, width, 3))

    # return the processed image
    return image


@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}
    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        # read the image in PIL format
        image = flask.request.files["file"].read()
        image = Image.open(io.BytesIO(image))

        image = prepare_image(image, target=(width, width))

        # preprocess the image and prepare it for classification
        predict = model.predict(image)
        label = ['Golden Retriever', 'Labrador Retriever', 'Kuvasz']
        result = label[np.argmax(predict)]
        data["result"] = str(result)

        # classify the input image and then initialize the list
        # of predictions to return to the client

        # indicate that the request was a success
        data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    app.run()
