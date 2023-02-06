import requests
import json
import cv2


url = "http://<external IP address>:5000/predict"
headers = {"content-type": "image/jpg"}

# encode image
raw_image_upload = cv2.imread('images/highway.jpg')
params = [cv2.IMWRITE_JPEG_QUALITY, 50,  cv2.IMWRITE_JPEG_OPTIMIZE, 1]
_, raw_image_encoded = cv2.imencode(".jpg", raw_image_upload,params)


# send HTTP request to the server
response = requests.post(url, data=img_encoded.tostring(), headers=headers)
predictions = response.json()


#annotate image with predictions

for pred in predictions:
    # print prediction
    print(pred)

    # extract the bounding box coordinates
    (x, y) = (pred["boxes"][0], pred["boxes"][1])
    (w, h) = (pred["boxes"][2], pred["boxes"][3])
    
    


    # draw a bounding box rectangle and label on the image
    cv2.rectangle(raw_image_upload, (x, y), (x + w, y + h), pred["color"], 2)
    text = "{}: {:.4f}".format(pred["label"], pred["confidence"])
    font_scale = min(w, h) * font_scale
    thickness = math.ceil(min(w, h) * thickness_scale)
    
    cv2.putText(
        raw_image_upload, 
        text, 
        (x, y - 5), 
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8, 
        pred["color"], 
        3
    )
    
#save annotated image
cv2.imwrite("annotated_image.jpg", raw_image_upload)
