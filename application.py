"""
PicMetric Flask API: Given a single images or batch
of images, analyzes the images and returns a JSON with
summary information about the image(s) and batch.
"""

from flask import Flask, jsonify, request, render_template
# import model_yolov3
import numpy as np  # [?? To do: REMOVE after removing testing part below ??]
import pickle # [?? To do: REMOVE after removing testing part below ??]
from sklearn.dummy import DummyClassifier # [?? To do: REMOVE after removing testing part below ??]
from sklearn.pipeline import make_pipeline # [?? To do: REMOVE after removing testing part below ??]
from sklearn.preprocessing import OrdinalEncoder # [?? To do: REMOVE after removing testing part below ??]


# Initialize our flask app (API):
application = Flask(__name__)

# Load our dummy model:
with open('model.pkl', 'rb') as file:
    MODEL = pickle.load(file)
    file.close()

# Base route just so AWS doesn't show status as problematic:
@application.route('/')
def root():
    return 'Welcome to the PicMetric machine learning API!'

# Route '/summary' that takes in a JSON with the URL for a
# user's image, analzyes the image with our ML (DL) model, and
# returns information about the image in a JSON:
@application.route('/summary', methods=['POST'])
def summary():
    """
    Analyzes 1 image and returns summary information
    about the image.
    """
    # Check to make sure we received a valid JSON with the request:
    if not request.json:
        return jsonify({"error": "no request received"})
    # Get incoming request with the image data:
    image_json = request.get_json(force=True)['image'] # add checks with request.is_json() (and try + the second one?)
    # Get image summary from our ML model, and return it in JSON form:
    return jsonify(get_summary(image_json))

# Route '/batch_img_summary' that takes in a JSON with the URLs
# for multiple images in a batch, analzyes the images with our
# ML (DL) model, and returns information about all of the
# images in a JSON doc:
# Route at /resetdb that clears and resets our database:
@application.route('/batch_img_summary', methods=['POST'])
def batch_img_summary():
    """
    Analyzes a batch of multiple images, and returns
    summary information about them in a JSON.
    """
    # Check to make sure we received a valid JSON with the request:
    if not request.json:
        return jsonify({"error": "no request received"})
    # Get incoming request with the image data:
    batch_images_jsons = request.get_json(force=True)['images'] # add checks with request.is_json() (and try + the second one?)

    image_summaries = []
    for image_json in batch_images_jsons:
        image_summaries.append(get_summary(image_json))

    # Get image summaries for all images in the batch from
    # our ML model, and return them in JSON form:
    return jsonify({"images": image_summaries})

# Function to get summary information for 1 single image
# from our ML image recognition model, and return it as
# a JSON:
def get_summary(image_json):
    """
    Function to get summary information for 1 single image
    from our yolov3 CNN DL model, and return it as a JSON.
    """
    image_id = image_json['image_id']
    image_url = image_json['image_url']

    # [?? To do: REMOVE below -- for testing only! ??]
    input_for_model = np.random.randint(0, 10, 1).reshape(-1, 1)

    people_present = int(MODEL.predict(input_for_model))

    image_summary = {
                     "image_id": image_id,
                     "image_url": image_url,
                     "image_url_marked_up": image_url,
                     "results":
                         [
                            {"prediction": "person",
                             "count": people_present,
                             "average_certainty": 100.0}
                         ]
                     }

    return image_summary

# While debugging:
if __name__ == "__main__":
    application.run(debug=False, port=8080) # [?? To do: Change debug to False ??]
