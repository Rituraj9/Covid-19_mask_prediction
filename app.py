import tensorflow as tf
from flask import Flask,request,jsonify,render_template,url_for,send_from_directory,Response
import cv2
from imutils.video import WebcamVideoStream
#making Single prediction
#from camera import VideoCamera
from keras.preprocessing import image
import numpy as np
import time

COUNT = 0

app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1

class VideoCamera(object):
	def __init__(self):
		self.stream = WebcamVideoStream(src=-1).start()
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

@app.route('/')
def man():
    return render_template('home.html')

@app.route('/videostream')
def pred():
    return render_template('detect.html') 

def gen(camera):
	while True:
		data = camera.get_frame()
		#print(image)
		#camera.runn(image)
		frame = data[0]
		yield(b'--frame\r\n'
			b'Content-Type: image/jpeg\r\n\r\n'+frame+b'\r\n\r\n')
		#pp(image)

@app.route("/video")
def video():
	return Response(gen(VideoCamera()),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/pic_upload")
def pic_upload():
	return render_template('pic.html')

@app.route('/pic_predict', methods=['POST'])
def pic_pred():
	mm = tf.keras.models.load_model('covid_model.pkl')
	#print(mm.summary())
	global COUNT
	img = request.files['image']
	img.save('static/{}.jpg'.format(COUNT))
	test_image = image.load_img('static/{}.jpg'.format(COUNT),target_size=(64,64))
	COUNT=COUNT+1
	test_image = image.img_to_array(test_image)
	#print(test_image)
	test_image = np.expand_dims(test_image,axis=0)
	#print(test_image)
	result = mm.predict(test_image)

	if result[0][0] == 1:
		predictions = 'Without mask'
	else:
		predictions = 'With Mask'

	#print(predictions)
	return render_template('prediction.html', data=predictions)

@app.route('/load_img')
def load_img():
    global COUNT
    print(COUNT)
    return send_from_directory('static', "{}.jpg".format(COUNT-1))

if __name__ == '__main__':
    app.run(debug=True)
