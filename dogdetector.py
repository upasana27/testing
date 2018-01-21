import numpy as np
from keras.preprocessing import image
#import qwe ourCNNcode
from keras.models import load_model
from keras.models import Sequential
import cv2
cap = cv2.VideoCapture(0)
model = Sequential()
#weightLoad
fname = 'first_try.h5'
model.load_weights(fname)

while(True):
    # Capture frame-by-frame
    	ret, frame = cap.read()
  test_image = image.load_img('images.jpeg', target_size = (150, 150))
	test_image = image.img_to_array(test_image)
	test_image = np.expand_dims(test_image, axis = 0)
	result = model.predict(test_image)

	if result[0][0] == 1:
 	prediction = 'dog'
	else:
 	prediction = 'road'
	print prediction
	cv2.waitkey(1)
