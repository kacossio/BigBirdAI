from PIL import Image
from numpy import asarray
import torchvision
import onnxruntime as rt
import numpy as np
import time
import torch
data = asarray(Image.open('/Users/devinuner/Desktop/yolov5-onnx-tensorrt/samples/0000001_02999_d_0000005.jpg')).reshape(1,3,640,640)
sess = rt.InferenceSession("/Users/devinuner/Desktop/yolov5/yolov5s.onnx")
input_name = sess.get_inputs()[0].name
pred_onx = sess.run(None, {input_name: data.astype(np.float32)})[0]

# preds = []
# for i,p in enumerate(pred_onx[0]):
#   if max(p) > 0.9:
#       print(i)

print(pred_onx.shape)
def non_max_suppression(prediction, conf_thres=0.5, iou_thres=0.6, classes=None, agnostic=False, labels=()):
    """
    Performs Non-Maximum Suppression (NMS) on inference results
    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """

    # Number of classes.
    nc = prediction[0].shape[1] - 5
    
    # Candidates.
    xc = prediction[..., 4] > conf_thres

    # Settings:
    # Minimum and maximum box width and height in pixels.
    min_wh, max_wh = 2, 4096

    # Maximum number of detections per image.
    max_det = 300
    
    #Timeout.
    time_limit = 10.0  
    
    # Require redundant detections.
    redundant = True
    
    # Multiple labels per box (adds 0.5ms/img).
    multi_label = nc > 1
    
    # Use Merge-NMS.
    merge = False

    t = time.time()
    output = [torch.zeros(0, 6)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        
        # Apply constraints:
        # Confidence.
        x = x[xc[xi]]

        # Cat apriori labels if autolabelling.
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image.
        if not x.shape[0]:
            continue

        # Compute conf.
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2).
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls).
        if multi_label:
            # print(torch.Tensor(x[:, 5:] > conf_thres))
            i, j = torch.Tensor(x[:, 5:] > conf_thres).nonzero(as_tuple=False).T.numpy()
            x = torch.cat((torch.Tensor(box[i]), torch.Tensor(x[i, j + 5, None]), torch.Tensor(j[:, None].astype(float))), 1)
        else:

            # Best class only.
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class.
        if classes:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # If none remain process next image.
        # Number of boxes.
        n = x.shape[0]
        if not n:
            continue

        # Batched NMS:
        #Classes.
        c = x[:, 5:6] * (0 if agnostic else max_wh)
        
        # Boxes (offset by class), scores.
        boxes, scores = x[:, :4] + c, x[:, 4]
        
        # NMS.
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        
        # Limit detections.
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):
            
            # Merge NMS (boxes merged using weighted mean).
            # Update boxes as boxes(i,4) = weights(i,n) * boxes(n,4).
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y
out = non_max_suppression(pred_onx, conf_thres=0.7)[0]

print(out.shape)