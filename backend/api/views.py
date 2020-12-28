import os
import json

from flask import Flask, request, redirect, render_template, jsonify
from uuid import uuid4

from backend.api.helpers import recognize_bird

img_upload_directory = './data/test2'

app = Flask(__name__)


@app.route('/api/bird/recognize', methods=['POST'])
def bird_recognize():
    """
    1. Save the uploaded file to the storage
    2. Trigger predictions
    3. Return predictions
    """
    bird = request.files['bird']
    new_name = str(uuid4())
    bird.save(os.path.join(img_upload_directory, new_name))

    data = recognize_bird(os.path.join(img_upload_directory, new_name))
    return jsonify(data)