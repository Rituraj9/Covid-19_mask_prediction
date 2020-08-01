import tensorflow as tf
from flask import Flask,request,jsonify,render_template,url_for,send_from_directory,Response
#making Single prediction
from camera import VideoCamera
from keras.preprocessing import image
import numpy as np
COUNT = 0
app = Flask(__name__)

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
