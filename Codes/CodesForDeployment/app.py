from flask import Flask, render_template, request
from keras.models import load_model
#import cv2
import matplotlib.image as mpimg
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K

app = Flask(__name__)

#Competition Matrix
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

model = load_model('UNET.h5',custom_objects={'dice_coef':dice_coef})

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/prediction", methods=['GET', 'POST'])
def prediction():
    
    img = request.files['img']
    img.save('img.jpg')
    imagePath = 'img.jpg'
    
    test_image = image.load_img(imagePath, target_size = (128,800))
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image)
    
    fig, axs = plt.subplots(4,2, figsize=(16,8))
    image_test = mpimg.imread(imagePath)
    for i in range(4):
        axs[i,0].imshow(image_test)
        axs[i,0].set_title('Test Image')
        axs[i,1].imshow(result[0,:,:,i])
        axs[i,1].set_title("Predicted mask for Class '{}'".format(i+1))
    plt.savefig('static/predictedImage.jpg')
    plt.show()
        
    
        
    return render_template('prediction.html')

if __name__ == "__main__":
    app.run(debug = True)
