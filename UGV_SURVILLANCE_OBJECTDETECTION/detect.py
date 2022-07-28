import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
import string
import random
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession



def predict(ima):
    model = 'yolov4'
    tiny = True
    framework = 'tf'
    weights = './checkpoints/yolov4-416'
    size = 416
    img=ima
    iou = 0.45
    score = 0.25
    dont_show = False
    output = 'static/detections/'

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    input_size = size
    images = img

    saved_model_loaded = tf.saved_model.load(weights, tags=[tag_constants.SERVING])

    # loop through images in list and run Yolov4 model on each
    for count, image_path in enumerate(images, 1):
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        image_data = cv2.resize(original_image, (input_size, input_size))
        image_data = image_data / 255.

        images_data = []
        for i in range(1):
            images_data.append(image_data)
        images_data = np.asarray(images_data).astype(np.float32)

        infer = saved_model_loaded.signatures['serving_default']
        batch_data = tf.constant(images_data)
        pred_bbox = infer(batch_data)
        for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=iou,
            score_threshold=score
        )
        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
        image = utils.draw_bbox(original_image, pred_bbox)
        image = Image.fromarray(image.astype(np.uint8))
        #if not dont_show:
            #image.show()
        res = ''.join(random.choices(string.ascii_uppercase +
                                     string.digits, k=7))
        url=(output +res+ '.png')
        image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        imgurl=cv2.imwrite(url, image)
        #print(classes)
        #print(valid_detections)
        q=np.unique(valid_detections)
        z=int(q)
        #print(type(z),z)
        unique=np.unique(classes)
        #print(unique)
        predclass=[]
        for a in range(z):
            if classes[0][a]==0:
                predclass.append("accident")
            elif classes[0][a]==1:
                predclass.append("fire")
            elif classes[0][a]==2:
                predclass.append("medical emergency")
            elif classes[0][a]==3:
                predclass.append("mob")
            elif classes[0][a]==4:
                predclass.append("smoke")
        predclass=np.array(predclass)
        preclass=np.unique(predclass)
        #print(preclass)
        return preclass,url


def listToString(s):
    str1 = ""
    for ele in s:
        str1+=ele+","
    strlen=len(str1)
    fstr=str1[0:strlen- 1]
    return fstr


def alertmsg(alt):
    s=[]
    for i in alt:
        if(i=="accident" or i=='medical emergency' ):
            s.append("   !!!  Please call for ambulance and alert the near by hosipital !!!")
        elif(i=="mob"):
            s.append("   !!!  Inform Police and call for rescue   !!!")
        elif(i=="fire"):
            s.append("   !!!  Alert the near by Fire station !!!")
        elif(i=="smoke"):
            s.append("   !!!   Alert the near by Fire station !!!")
    return s