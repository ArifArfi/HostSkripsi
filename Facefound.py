import cv2
import time
import base64
import uuid
from datetime import datetime
from paho.mqtt import client as mqtt_client
import json
import requests




base_url = "http://127.0.0.1/"
topic = "camera/photo"


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to broker")
        global connected
        connected = True
        print(connected)
    else:
        print("Failed to connect")


connected = False
time_millis = 0
delay = 10

client = mqtt_client.Client("my_client_" + str(uuid.uuid4().hex))
client.on_connect = on_connect
client.subscribe(topic)
client.connect("127.0.0.1", 1883)
client.loop_start()

print(connected)

while not connected:
    print(connected)
    time.sleep(0.1)

received_data = None

def on_message(client, userdata, message):
    global received_data
    payload = message.payload.decode('utf-8')
    try:
        data_json = json.loads(payload)
        received_data = data_json  # Store the JSON data in the global variable
        # print(f"test {data_json}")
    except json.JSONDecodeError:
        print("Invalid JSON format")
client.on_message = on_message
client.subscribe("face/recognition")  # Replace "camera/json_data" with the topic containing the JSON data



# https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cv2.namedWindow("img", cv2.WINDOW_GUI_EXPANDED)
cap = cv2.VideoCapture(0)
face_images = []
# captured = 0


def publish(client):
    msg_count = 1
    for face in face_images:
        # Convert face image to base64 string
        _, buffer = cv2.imencode('.jpg', face)
        face_base64 = base64.b64encode(buffer).decode('utf-8')
        # Prepare MQTT message
        msg = f"Face {msg_count}: {face_base64}"
        result = client.publish(topic, msg)
        # result: [0, 1]
        status = result[0]
        if status == 0:
            print(f"Sent Face {msg_count} to topic `{topic}`")
        else:
            print(f"Failed to send Face {msg_count} to topic {topic}")
        msg_count += 1

# captured = 1
prediksi_label = ""
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    print(prediksi_label)
    # print(ret)
    if frame is None:
        print("No Frame")
        continue

    # frame = cv2.imread("D:\\New folder\\face_recog_dlib_file\Dataset2\Arif\Arif_3.jpg")
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        if len(faces) > 0:
            crop = frame.copy()[y:y + h, x:x + w]
            resize = cv2.resize(crop, (50, 50), interpolation=cv2.INTER_LINEAR)
            encode = cv2.imencode(".jpg", resize)[1]
            img_str = base64.b64encode(encode.tobytes())
            # print(img_str.decode())
            # cv2.imwrite("dataset/" + datetime.now().strftime('%Y%m%d%H%M%S%f') + ".jpg", resize)
            current_seconds = round(time.time() * 1000) / 1000
            if (current_seconds - time_millis) > delay:
                labels_device = ["1"]
                dt_payload = (labels_device[0] + "_" + img_str.decode()).encode('utf-8')
                client.publish(topic="camera/photo", payload=dt_payload, qos=0)
                time_millis = current_seconds
                # Check if there's data received from MQTT
                if received_data is not None:
                    try:
                        openjson = received_data
                        label_y = y - 10  # Adjust the vertical position of the label above the rectangle
                        prediksi_label = openjson['label']
                        print(f"terprediksi {openjson['label']} dengan akurasi {openjson['persentase']}")
                        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Menggunakan datetime untuk mendapatkan waktu terkini
                        data_jsons = {
                            "data": str(img_str, 'utf-8'),
                            "device_id": labels_device,
                            "waktu": now
                        }
                        json_result = json.dumps(data_jsons, indent=4)
                        print(json_result)
                    except Exception as e:
                        print(e)
                        pass
            cv2.putText(frame, str(prediksi_label), (x, y - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1.5,
                        (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow('img', frame)
    time.sleep(0.25)
    if cv2.waitKey(1) & 0xff == ord('s'):
        break
cap.release()
cv2.destroyAllWindows()
client.loop_stop()