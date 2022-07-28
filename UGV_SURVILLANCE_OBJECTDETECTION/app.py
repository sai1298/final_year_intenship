import os
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from detect import predict,listToString,alertmsg
# Define a flask app
from flask import jsonify
app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('main.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    rest=0
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'static/uploads', secure_filename(f.filename))
        f.save(file_path)
        # Make prediction
        path = []
        path.append(file_path)
        res,imgs = predict(path)
        result = listToString(res)
        aler=alertmsg(res)
        alert=listToString(aler)
        print(alert)
    return jsonify({
        "preclass":result,
        "imgurl":imgs,
        "alert":alert
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000, debug=True)

