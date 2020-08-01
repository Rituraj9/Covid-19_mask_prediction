import cv2
from imutils.video import WebcamVideoStream
import tensorflow as tf
import numpy as np
import time

class VideoCamera(object):
	def __init__(self):
		self.stream = WebcamVideoStream(src=0).start()
		time.sleep(2.0)

	def __del__(self):
		self.stream.stop()

	...

	def get_frame(self):
		image = self.stream.read()
		mm = tf.keras.models.load_model('covid_model.pkl')
		face_clsfr =cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
		color_dict={1:(0,0,255),0:(0,255,0)}
		faces=face_clsfr.detectMultiScale(image,1.3,5)
		for (x,y,w,h) in faces:
			face_img=image[y:y+w,x:x+w]
			resized=cv2.resize(face_img,(64,64))
			face_image = np.expand_dims(resized,axis=0)
			result=mm.predict(face_image)
			pred = result[0][0]>0.5
			pred = int(pred==True)
			print(pred)
			if pred==1:
				prediction='No Mask Detected'
			else:
				prediction='Mask Detected'

			cv2.rectangle(image,(x,y),(x+w,y+h),color_dict[pred],2)
			cv2.rectangle(image,(x,y-40),(x+w,y),color_dict[pred],-1)
			cv2.putText(image, prediction , (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

		ret,jpeg = cv2.imencode('.jpg',image)
		data=[]
		data.append(jpeg.tobytes())
		return data

	def runn(img):
		print(img)




