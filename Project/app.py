import matplotlib
matplotlib.use('Agg')

from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import io

app = Flask(__name__)

data = np.load("faces.npy")
target = np.load("faces_target.npy")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/show_40_distinct_people')
def show_40_distinct_people():
    fig, axarr = plt.subplots(nrows=4, ncols=10, figsize=(18, 9))
    axarr = axarr.flatten()
    unique_ids = np.unique(target)
    for unique_id in unique_ids:
        image_index = unique_id * 10
        axarr[unique_id].imshow(data[image_index], cmap='gray')
        axarr[unique_id].set_xticks([])
        axarr[unique_id].set_yticks([])
        axarr[unique_id].set_title("face id:{}".format(unique_id))
    plt.suptitle("There are 40 distinct people in the dataset")

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return send_file(buf, mimetype='image/png')

@app.route('/show_10_faces_of_n_subject')
def show_10_faces_of_n_subject():
    subject_id = int(request.args.get('subject_id', 0))
    fig, axarr = plt.subplots(nrows=1, ncols=10, figsize=(18, 9))
    for j in range(10):
        image_index = subject_id * 10 + j
        axarr[j].imshow(data[image_index], cmap="gray")
        axarr[j].set_xticks([])
        axarr[j].set_yticks([])
        axarr[j].set_title("face id:{}".format(subject_id))
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return send_file(buf, mimetype='image/png')

@app.route('/pca')
def pca():
    X = data.reshape((data.shape[0], data.shape[1] * data.shape[2]))
    X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.3, stratify=target, random_state=0)
    n_components = 90
    pca = PCA(n_components=n_components, whiten=True)
    pca.fit(X_train)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(pca.mean_.reshape((64, 64)), cmap="gray")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Average Face')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return send_file(buf, mimetype='image/png')


@app.route('/bar_graph')
def bar_graph():
    unique_ids, counts = np.unique(target, return_counts=True)
    plt.figure(figsize=(10, 6))
    plt.bar(unique_ids, counts, color='skyblue')
    plt.xlabel('Face ID')
    plt.ylabel('Number of Images')
    plt.title('Number of Images per Face ID')
    plt.xticks(unique_ids)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return send_file(buf, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
