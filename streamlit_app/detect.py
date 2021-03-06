import streamlit as st

import torch.backends.cudnn as cudnn
import torchvision.models as models
from torchvision import datasets, transforms, utils
from efficientnet_pytorch import EfficientNet

from models.experimental import *
from utils.datasets import *
from utils.utils import *

import numpy as np
from PIL import Image
import copy
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--weights', nargs='+', type=str, default='weights/best_lesion_yolov5l.pt', help='model.pt path(s)')
parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
# parser.add_argument('--conf-thres', type=float, default=0.55, help='object confidence threshold')
# parser.add_argument('--iou-thres', type=float, default=0.3, help='IOU threshold for NMS')
parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
opt = parser.parse_args(args = []) ##

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

#####################################
# Prepare image for YOLO
#####################################
def prepare_image(image_array):
    """
    Input: image_array
    Output: image resized to 512 and rescaled to 0-1. Shape (1, )
    """
    # Padded resize
    img = letterbox(image_array, new_shape=opt.img_size)[0]
    img = img.transpose(2, 0, 1)
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # (0 - 255) to (0.0 - 1.0)
    if img.ndimension() == 3:
        img = img.unsqueeze(0) # add dim for batch size 1 --> torch.Size([1, 3, 384, 512])
    return img
    
#####################################
# Second stage classifier functions
#####################################
def load_classifier(name='efficientnetb4',  device='cuda:0'):
    """
    Loads a second stage classifier to classify yolo bb outputs
    """
    if name=="densenet161":
        # model = models.__dict__[name](pretrained=False)
        model = models.densenet161(pretrained=False)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, 3) ### number of classes in dataset
        test_model_checkpoint=glob.glob("weights/DenseNet-161*_best.pt")[0]
    
    elif name=="efficientnetb4":
        model = EfficientNet.from_name('efficientnet-b4', num_classes=3) #, image_size=380)
        test_model_checkpoint=glob.glob("weights/EfficientNet-b4*_best.pt")[0]

    checkpoint = torch.load(test_model_checkpoint, map_location=torch.device(device))
    model.load_state_dict(checkpoint["model_state_dict"])
    return model
    
class SquarePad:
	def __call__(self, image):
		w, h = image.size
		max_wh = np.max([w, h])
		hp = int((max_wh - w) / 2)
		vp = int((max_wh - h) / 2)
		padding = (hp, vp, hp, vp)
		return transforms.functional.pad(image, padding, 0, 'constant') 

def classifier_img_transform(img, model_name):
    if model_name=="efficientnetb4":
        resize_to = (380, 380)
    elif model_name=="densenet161":
        resize_to = (224, 224)
        
    trnsfm = transforms.Compose([
        SquarePad(),
        transforms.Resize(resize_to, interpolation=3), 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # imagenet mean and std
    ]) 
    image_tensor = trnsfm(img).float()
    return image_tensor
    

def apply_classifier(x, model, model_name, img, im0):
    """
    Applies a second stage classifier to yolo bb outputs.
    Rescales the predicted box coordinates from img_size to the original size, then does the cut out before making class predictions
    Output: predicted class probabilities
    """
    classes = ["Benign", "Cancerous", "Precancerous"]
    
    
    im0 = [im0] if isinstance(im0, np.ndarray) else im0
    for i, d in enumerate(x):  # per image
        if d is not None and len(d):
            d = d.clone()

            # # Reshape and pad cutouts
            # st.write("Detector: xyxy coordinates - before square conv:", str(d[:, :4]))
            b = xyxy2xywh(d[:, :4])  # boxes
            # b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # rectangle to square
            b[:, 2:] = b[:, 2:] * 1.1 + 10  # pad
            d[:, :4] = xywh2xyxy(b).long()

            # Rescale box coordinates (d) from resized image shape HxW (i.e. 384x512) to im0 size: 
            scale_coords(img.shape[2:], d[:, :4], im0[i].shape)

            # Classes
            # pred_cls1 = d[:, 5].long() # lesion or not
            output_clf = []
            for j, a in enumerate(d):  # per item
                cutout = im0[i][int(a[1]):int(a[3]), int(a[0]):int(a[2])]
                # st.image(Image.fromarray(np.uint8(cutout)).convert('RGB'), use_column_width=True) # show cropped region
                # cv2.imwrite('test%i.jpg' % j, cutout)

                # convert the cutout to PIL format
                im = cv2.cvtColor(cutout, cv2.COLOR_BGR2RGB)
                im = Image.fromarray(im)

                pred_cls2 = model(classifier_img_transform(im, model_name).unsqueeze(0).to(d.device)) #clf probabilities 
                # pred_cls2 = model(classifier_img_transform(im, model_name).unsqueeze(0).to(d.device)).argmax(1) # classifier prediction
                # pred_cls2 = model(torch.Tensor(ims).to(d.device))  # classifier probabilities

                pred = pred_cls2[0].unsqueeze(0)
                prob = torch.nn.functional.softmax(pred, dim=1)[0] * 100
                _, indices = torch.sort(pred, descending=True)
                output_clf.append([(classes[idx], prob[idx].item()) for idx in indices[0][:3]])
            
    return output_clf
    
#####################################
# YOLOv5 detector
#####################################   
def yolov5(byte_image=None, conf_thres=0.55, overlap_thres=0.3, classify_bb=False, augment=False):
    """
    input: byte_image from streamlit.file_uploader
    output: image_array that contain predicted bounding boxes and labels
    """
    # st.write("Current device: ", device)

    # Load model
    model = attempt_load(opt.weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(opt.img_size, s=model.stride.max())  # check img_size
    model.to(device).eval()
    
    # Get names and colors
    # names = model.module.names if hasattr(model, 'module') else model.names
    names = ['lesion'] ##
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    
    # Second-stage classifier
    if classify_bb:
        clf_name = "efficientnetb4"
        modelc = load_classifier(name=clf_name, device=device)
        modelc.to(device).eval()


    # st.file_uploader returns BytesIO or StringIO or list of BytesIO/StringIO
    image_PIL = Image.open(byte_image).convert('RGB') # [:,:,:3]
    image_array = np.array(image_PIL) 
    img = prepare_image(image_array) # torch.Size([1, 3, 384, 512])

    # Predict
    pred = model(img, augment)[0]

    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, overlap_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

    # Process detections
    output=copy.deepcopy(image_array)
    if ((pred[0] is not None) and (len(pred[0])>0)):
        det = pred[0].clone() ### clone essential, otherwise pred changes due to changes to det below
        # st.write("Number of detected lesions: {}".format(len(det) if det!=None else "0"))
    
        # scale predicted bb coordinates to the original image size
        # st.write("YOLO: xyxy coordinates - before scaling:", str(det[:, :4]))
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], output.shape).round()

        # Write results on to the original image (image_array or output)
        for i, (*xyxy, conf, cls) in enumerate(det,1):
            label = '%s%i: %.2f' % (names[int(cls)], i, conf)
            plot_one_box(xyxy, output, label=label, color=colors[int(cls)], line_thickness=3)

        # Output both bb detections and their classes 
        if (classify_bb==True):
            output_clf = apply_classifier(pred, modelc, clf_name, img, image_array)
            return output, output_clf
        # Output only bb detections
        else:
            return output, None
    
    else:
        return output, None
