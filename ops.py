import numpy as np
from typing import Tuple, List, Optional
import cv2
import logging
import time
import os

file_log = logging.FileHandler('inference.log')
console_out = logging.StreamHandler()

logging.basicConfig(
    handlers=(file_log, console_out),
    datefmt='%m.%d.%Y %H:%M:%S',
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    level=logging.INFO,
)

LOGGER = logging.getLogger(__name__)


def letterbox(
    img: np.ndarray,
    new_shape: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
    auto: bool = False,
    scale_fill: bool = False,
    scaleup: bool = False,
    stride: int = 32,
) -> Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]:
    """
    Resize image and padding for detection. Takes image as input,
    resizes image to fit into new shape with saving original aspect ratio and pads it to meet stride-multiple constraints

    Parameters:
    img (np.ndarray): image for preprocessing
    new_shape (Tuple(int, int)): image size after preprocessing in format [height, width]
    color (Tuple(int, int, int)): color for filling padded area
    auto (bool): use dynamic input size, only padding for stride constrins applied
    scale_fill (bool): scale image to fill new_shape
    scaleup (bool): allow scale image if it is lower then desired input size, can affect model accuracy
    stride (int): input padding stride
    Returns:
    img (np.ndarray): image after preprocessing
    ratio (Tuple(float, float)): hight and width scaling ratio
    padding_size (Tuple(int, int)): height and width padding size

    """
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scale_fill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def processing_image(
    image: np.ndarray,
    new_shape: Tuple[int, int],
    half: bool = False,
) -> np.ndarray:
    """
    Preprocess image according to YOLOv8 input requirements.
    Takes image in np.array format, resizes it to specific size using letterbox resize and changes data layout from HWC to CHW.

    Parameters:
    image (np.ndarray): image for preprocessing
    new_shape (tuple(int, int)): shape to resize and pad
    Returns:
    img (np.ndarray): image after preprocessing
    """

    image = letterbox(image, new_shape=new_shape)[0]
    image = image.transpose(2, 0, 1)
    image = np.ascontiguousarray(image)
    
    input_tensor = image.astype(np.float16 if half else np.float32)
    input_tensor /= 255.0
    if input_tensor.ndim == 3:
        input_tensor = np.expand_dims(input_tensor, 0)
    return input_tensor


def nms(
    boxes,scores: np.ndarray,
    overlap_threshold: float = 0.5,
    min_mode=False,
) -> np.ndarray:
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = scores

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    index_array = scores.argsort()[::-1]
    keep = []
    while index_array.size > 0:
        keep.append(index_array[0])
        x1_ = np.maximum(x1[index_array[0]], x1[index_array[1:]])
        y1_ = np.maximum(y1[index_array[0]], y1[index_array[1:]])
        x2_ = np.minimum(x2[index_array[0]], x2[index_array[1:]])
        y2_ = np.minimum(y2[index_array[0]], y2[index_array[1:]])

        w = np.maximum(0.0, x2_ - x1_ + 1)
        h = np.maximum(0.0, y2_ - y1_ + 1)
        inter = w * h

        if min_mode:
            overlap = inter / np.minimum(areas[index_array[0]], areas[index_array[1:]])
        else:
            overlap = inter / (areas[index_array[0]] + areas[index_array[1:]] - inter)

        inds = np.where(overlap <= overlap_threshold)[0]
        index_array = index_array[inds + 1]

    return keep


def non_max_suppression(
    prediction: np.ndarray,
    conf_thres: float = 0.25,
    iou_thres: float = 0.7,
    classes=None,
    agnostic: bool = False,
    multi_label: bool = False,
    labels=(),
    max_det: int = 300,
    nc: int = 0,  # number of classes (optional)
    max_time_img: float = 0.05,
    max_nms: int = 30000,
    max_wh: int = 7680,
) -> List[np.ndarray]:   
    """
    Perform non-maximum suppression (NMS) on a set of boxes, with support for masks and multiple labels per box.

    Arguments:
        prediction (numpy array): A numpy array of shape (batch_size, num_classes + 4 + num_masks, num_boxes)
            containing the predicted boxes, classes, and masks. The tensor should be in the format
            output by a model, such as YOLO.
        conf_thres (float): The confidence threshold below which boxes will be filtered out.
            Valid values are between 0.0 and 1.0.
        iou_thres (float): The IoU threshold below which boxes will be filtered out during NMS.
            Valid values are between 0.0 and 1.0.
        classes (List[int]): A list of class indices to consider. If None, all classes will be considered.
        agnostic (bool): If True, the model is agnostic to the number of classes, and all
            classes will be considered as one.
        multi_label (bool): If True, each box may have multiple labels.
        labels (List[List[Union[int, float, numpy array]]]): A list of lists, where each inner
            list contains the apriori labels for a given image. The list should be in the format
            output by a dataloader, with each label being a tuple of (class_index, x1, y1, x2, y2).
        max_det (int): The maximum number of boxes to keep after NMS.
        nc (int, optional): The number of classes output by the model. Any indices after this will be considered masks.
        max_time_img (float): The maximum time (seconds) for processing one image.
        max_nms (int): The maximum number of boxes into nms().
        max_wh (int): The maximum box width and height in pixels

    Returns:
        (List[numpy array]): A list of length batch_size, where each element is a tensor of
            shape (num_boxes, 6 + num_masks) containing the kept boxes, with columns
            (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
    """
    if isinstance(prediction, (list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output
    output = []
    bs = prediction.shape[0]  # batch size
    nc = nc or (prediction.shape[1] - 4)
    nm = prediction.shape[1] - nc - 4
    mi = 4 + nc  # mask start index
    xc = np.max(prediction[:, 4:mi], axis=1) > conf_thres  # candidates

    # Settings
    min_wh = 2  # (pixels) minimum box width and height
    time_limit = 0.5 + max_time_img * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS
    # print("shape of prediction", prediction.shape)

    prediction = np.transpose(prediction, (0, 2, 1))
    prediction = np.concatenate((xywh2xyxy(prediction[..., :4]), prediction[..., 4:]), axis=-1)  # xywh to xyxy

    output = [np.zeros((0, 6 + nm))] * bs
    t = time.time()
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = np.zeros((len(lb), nc + nm + 4))
            v[:, :4] = xywh2xyxy(lb[:, 1:5])  # box
            v[np.arange(len(lb)), lb[:, 0].astype(int) + 4] = 1.0  # cls
            x = np.concatenate((x, v), axis=0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls, mask = np.split(x, (4, 4 + nc, ), axis=1)

        if multi_label:
            i, j = np.where(cls > conf_thres)
            x = np.concatenate((box[i], x[i, 4 + j, None], j[:, None].astype(float), mask[i]), axis=1)
        else:  # best class only
            conf = np.max(cls,axis=1)
            conf = conf.reshape(conf.shape[0], 1)
            j = np.argmax(cls[:,:], axis=1, keepdims=True)
            x = np.concatenate((box, conf, j, mask), axis = 1)

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == np.array(classes, dtype=x.dtype)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        if n > max_nms:  # excess boxes
            x = x[np.argsort(x[:, 4])[::-1][:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        scores = x[:, 4]  # scores
        boxes = x[:, :4] + c  # boxes (offset by class)
        i = nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            LOGGER.warning(f'NMS time limit {time_limit:.3f}s exceeded')
            break  # time limit exceeded

    return output


def clip_boxes(
    boxes: np.ndarray,
    shape: Tuple[int, int],
) -> None:
    """
    Takes a list of bounding boxes and a shape (height, width) and clips the bounding boxes to the shape.

    Args:
    boxes (torch.Tensor): the bounding boxes to clip
    shape (tuple): the shape of the image
    """
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2


def scale_boxes(
    img1_shape: Tuple[int, int],
    boxes: np.ndarray,
    img0_shape: Tuple[int, int],
    ratio_pad: Tuple[float, float] = None,
    padding: bool = True,
) -> np.ndarray:
    """
    Rescales bounding boxes (in the format of xyxy) from the shape of the image they were originally specified in
    (img1_shape) to the shape of a different image (img0_shape).

    Args:
        img1_shape (tuple): The shape of the image that the bounding boxes are for, in the format of (height, width).
        boxes (torch.Tensor): the bounding boxes of the objects in the image, in the format of (x1, y1, x2, y2)
        img0_shape (tuple): the shape of the target image, in the format of (height, width).
        ratio_pad (tuple): a tuple of (ratio, pad) for scaling the boxes. If not provided, the ratio and pad will be
            calculated based on the size difference between the two images.
        padding (bool): If True, assuming the boxes is based on image augmented by yolo style. If False then do regular
            rescaling.

    Returns:
        boxes (torch.Tensor): The scaled bounding boxes, in the format of (x1, y1, x2, y2)
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1), round(
            (img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1)  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    if padding:
        boxes[..., [0, 2]] -= pad[0]  # x padding
        boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)

    return boxes


def xywh2xyxy(x: np.ndarray) -> np.ndarray:
    """
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner.

    Args:
        x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.
    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


class PrePostProcessor:
    def __init__(
        self,
        cls_names: Tuple[str],
        imgsz: tuple,
        conf: float,
        iou: float,
        half: bool = False,
        agnostic: bool = False,
        max_det: int = 300,
    ):
        """
        Creates pre/post process for inference
        Inputs:
            cls_names - names of classes
            imgsz - tuple of (h, w)
            conf - min conf score
            iou - min iou score
            half - convert image in np.float16 or not
            agnostic - agnostic nms or not
            max_det - max det in nms
        """
        if not (0 <= conf <= 1):
            LOGGER.error(f'Invalid Confidence threshold {conf}, valid values are between 0.0 and 1.0')
        if not (0 <= iou <= 1):
            LOGGER.error(f'Invalid IoU {iou}, valid values are between 0.0 and 1.0')

        self.cls_names = cls_names
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.half = half
        self.agnostic = agnostic
        self.max_det = max_det
        
    def postprocess(
        self,
        preds_boxes: np.ndarray,
        orig_imgs: List[np.ndarray],
    ) -> List[np.ndarray]:
        
        preds = non_max_suppression(
            preds_boxes,
            conf_thres=self.conf,
            iou_thres=self.iou,
            nc=len(self.cls_names),
            max_det=self.max_det,
            agnostic=self.agnostic,
        )
        
        results = []
        for i, pred in enumerate(preds):
            if not len(pred):
                results.append({"det": []})
                continue
            pred[:, :4] = scale_boxes(self.imgsz, pred[:, :4], orig_imgs[i].shape).round()
            results.append({"det": pred})

        return results
    
    @staticmethod
    def load_image(img_path: str) -> np.ndarray:
        if not os.path.exists(img_path):
            LOGGER.error(f"Image not found by path {img_path}. \n")
            
        # Read the input image using OpenCV
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        
        return img
        
    def preprocess(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Preprocess image according to YOLOv8 input requirements.
        Takes image in np.array format, resizes it to specific size using
        letterbox resize and changes data layout from HWC to CHW.

        Parameters:
            image (np.ndarray): image for preprocessing in BGR format
        Returns:
            img (np.ndarray): image after preprocessing
        """
        image, ratio, (dw, dh) = letterbox(image, new_shape=self.imgsz)
        image = image.transpose(2, 0, 1)
        image = np.ascontiguousarray(image)

        input_tensor = image.astype(np.float16 if self.half else np.float32)
        input_tensor /= 255.0

        if input_tensor.ndim == 3:
            input_tensor = np.expand_dims(input_tensor, 0)
            
        return input_tensor
    
    def plot(
        self,
        image: np.ndarray,
        bboxes: np.ndarray,
        save_path: str,
    ) -> np.ndarray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        for bbox in bboxes:
            xmin, ymin, xmax, ymax, score, cls_id = bbox
            xmin, ymin, xmax, ymax, score, cls_name = int(xmin), int(ymin), int(xmax), int(ymax), float(score), self.cls_names[int(cls_id)]

            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(image, f'{cls_name}', (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imwrite(save_path, image)
        LOGGER.info(f"Saving image_bbox_debug to {save_path}")