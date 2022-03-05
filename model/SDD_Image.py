import numpy as np
import time
import cv2
import math

# path specification 
labelsPath = "/home/jai/Desktop/projects/social-distancing-detection/Mini-project-Social-distance-detector/coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

weightsPath = "/home/jai/Desktop/projects/yolov4.weights"
configPath = "/home/jai/Desktop/projects/social-distancing-detection/Mini-project-Social-distance-detector/yolov4.cfg"

# reading darknet neural network w.r.t config and weights path 
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# reading image and processing the frames for distance computation 
image =cv2.imread('/home/jai/Desktop/projects/social-distancing-detection/Mini-project-Social-distance-detector/images/test_image_3.jpg')
(H, W) = image.shape[:2]
frameWidth = 960
frameHeight = 810
frameSize = (frameWidth,frameHeight)
imageResized = cv2.resize(image,frameSize)
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
blob = cv2.dnn.blobFromImage(imageResized, 1 / 255.0, (416, 416),swapRB=True, crop=False)
net.setInput(blob)
start = time.time()
layerOutputs = net.forward(ln)
end = time.time()
print("Frame Prediction Time : {:.6f} seconds".format(end - start))
boxes = []
confidences = []
classIDs = []
# localizing person objects with bounding boxes
for output in layerOutputs:
    for detection in output:
        scores = detection[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]
        if confidence > 0.5 and classID == 0:
            box = detection[0:4] * np.array([W, H, W, H])
            (centerX, centerY, width, height) = box.astype("int")
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))
            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            classIDs.append(classID)
            
idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,0.3)
ind = []
for i in range(0,len(classIDs)):
    if(classIDs[i]==0):
        ind.append(i)
a = []
b = []
color = (0,255,0) 
if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            a.append(x)
            b.append(y)
            cv2.rectangle(imageResized, (x, y), (x + w, y + h), color, 2)
            

distance=[] 
nsd = []
for i in range(0,len(a)-1):
    for k in range(1,len(a)):
        if(k==i):
            break
        else:
            x_dist = (a[k] - a[i])
            y_dist = (b[k] - b[i])
            d = math.sqrt(x_dist * x_dist + y_dist * y_dist)
            distance.append(d)
            if(d<=100.0):
                nsd.append(i)
                nsd.append(k)
            nsd = list(dict.fromkeys(nsd))
   
color = (0, 0, 255)
text=""
for i in nsd:
    (x, y) = (boxes[i][0], boxes[i][1])
    (w, h) = (boxes[i][2], boxes[i][3])
    cv2.rectangle(imageResized, (x, y), (x + w, y + h), color, 2)
    text = "Alert"
    cv2.putText(imageResized, text, (x, y - 5), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,0.5, color, 2)
           
cv2.putText(imageResized, text, (x, y - 5), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,0.5, color, 2)
cv2.imshow("Social Distancing Detector", imageResized)
cv2.imwrite('output.jpg', image)
cv2.waitKey()
cv2.destroyAllWindows()
