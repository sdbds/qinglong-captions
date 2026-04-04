import math
import os.path as osp
from enum import IntEnum

import numpy as np
from typing import List, Union
import cv2
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image, ImageDraw, ImageFont
import colorsys

from .cv import rgba_to_rgb_fixbg, img_alpha_blending
from .io_utils import imglist2imgrid, load_image


def show_factorization_on_image(img: np.ndarray,
                                explanations: np.ndarray,
                                colors: List[np.ndarray] = None,
                                image_weight: float = 0.5,
                                concept_labels: List = None,
                                visible_mask=None,) -> np.ndarray:
    """
    Modified from pytorch_grad_cam.utils.image.show_factorization_on_image

    Color code the different component heatmaps on top of the image.
        Every component color code will be magnified according to the heatmap itensity
        (by modifying the V channel in the HSV color space),
        and optionally create a lagend that shows the labels.

        Since different factorization component heatmaps can overlap in principle,
        we need a strategy to decide how to deal with the overlaps.
        This keeps the component that has a higher value in it's heatmap.

    :param img: The base image RGB format.
    :param explanations: A tensor of shape num_componetns x height x width, with the component visualizations.
    :param colors: List of R, G, B colors to be used for the components.
                   If None, will use the gist_rainbow cmap as a default.
    :param image_weight: The final result is image_weight * img + (1-image_weight) * visualization.
    :concept_labels: A list of strings for every component. If this is paseed, a legend that shows
                     the labels and their colors will be added to the image.
    :returns: The visualized image.
    """
    concept_per_pixel = explanations.argmax(axis=0)
    counts_ids = np.unique(concept_per_pixel)
    counts_ids = list(c for c in counts_ids)

    n_components = len(counts_ids)
    if colors is None:
        # taken from https://github.com/edocollins/DFF/blob/master/utils.py
        _cmap = plt.cm.get_cmap('gist_rainbow')
        colors = [
            np.array(
                _cmap(i)) for i in np.arange(
                0,
                1,
                1.0 /
                n_components)]

    masks = []
    for ii, counts_id in enumerate(counts_ids):
        mask = np.zeros(shape=(img.shape[0], img.shape[1], 3))
        mask[:, :, :] = colors[ii][:3]
        explanation = explanations[counts_id]
        # explanation[concept_per_pixel != i] = 0
        explanation = concept_per_pixel == counts_id
        mask = np.uint8(mask * 255)
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2HSV)
        mask[:, :, 2] = np.uint8(255 * explanation)
        mask = cv2.cvtColor(mask, cv2.COLOR_HSV2RGB)
        mask = np.float32(mask) / 255
        masks.append(mask)

    mask = np.sum(np.float32(masks), axis=0)
    result = img * image_weight + mask * (1 - image_weight)
    result = np.uint8(result * 255)
    if visible_mask is not None:
        result = result * visible_mask + np.full_like(result, fill_value=255) * (1 - visible_mask)
        result = result.astype(np.uint8)

    if concept_labels is not None:
        px = 1 / plt.rcParams['figure.dpi']  # pixel in inches
        fig = plt.figure(figsize=(result.shape[1] * px, result.shape[0] * px))
        plt.rcParams['legend.fontsize'] = int(
            20 * result.shape[0] / 256 / max(1, n_components / 6))
        lw = 5 * result.shape[0] / 256
        lines = [Line2D([0], [0], color=colors[i], lw=lw)
                 for i in range(n_components)]
        c_labels = [concept_labels[i] for i in counts_ids]
        plt.legend(lines,
                   c_labels,
                   mode="expand",
                   fancybox=True,
                   shadow=True)

        plt.tight_layout(pad=0, w_pad=0, h_pad=0)
        plt.axis('off')
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        plt.close(fig=fig)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        data = cv2.resize(data, (result.shape[1], result.shape[0]))
        result = np.hstack((result, data))
    return result


def visualize_segs_with_labels(mask_list, src_img: np.ndarray, tag_list, image_weight=0.3, colors=None, reference_img: np.ndarray = None, draw_legend=True) -> np.ndarray:

    n_components = len(mask_list)

    img = rgba_to_rgb_fixbg(src_img)

    colors = np.array([get_color(ii) for ii in range(n_components)], dtype=np.uint8)
    
    colored_list = [img]
    for ii, mask in enumerate(mask_list):
        color = np.array(colors[ii][:3])
        mask = mask.astype(np.float32)
        c = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
        c[..., :3] = (color[None, None] * mask[..., None]).astype(np.uint8)
        c[..., -1] = np.clip(mask.astype(np.float32) * 255 * (1 - image_weight), 0, 255).astype(np.uint8)
        colored_list.append(c)

    colored_final = img_alpha_blending(colored_list)
    result = colored_final[..., :3]

    if not draw_legend or tag_list is None:
        if reference_img is not None:
            result = np.hstack((reference_img, result))
        return result

    colors = colors.astype(np.float32) / 255.
    px = 1 / plt.rcParams['figure.dpi']  # pixel in inches
    fig = plt.figure(figsize=(result.shape[1] * px, result.shape[0] * px), facecolor=[0, 0, 0, 0])
    
    fnt_sz = int(5 * result.shape[0] / 256)
    plt.rcParams['legend.fontsize'] = fnt_sz
    lw = 5 * result.shape[0] / 256
    lines = [Line2D([0], [0], color=colors[i], lw=lw)
                for i in range(n_components)]
    # c_labels = [all_labels[i] for i in all_labels]
    plt.legend(lines,
                tag_list,
                mode="expand",
                fancybox=False,
                edgecolor="black",
                # frameon=False,
                shadow=False,
                framealpha=0.)

    plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    plt.axis('off')
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.buffer_rgba() , dtype=np.uint8)
    plt.close(fig=fig)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    dx, dy, dw, dh = cv2.boundingRect(cv2.findNonZero(data[..., 3]))
    
    data = rgba_to_rgb_fixbg(data[:, dx: dx + dw])
    data = cv2.copyMakeBorder(data, 0, 0, fnt_sz, fnt_sz, borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255))
    if data.shape[0] != img.shape[0]:
        data = cv2.resize(data, (dw, result.shape[0]))

    if reference_img is not None:
        result = np.hstack((reference_img, result, data))
    else:
        result = np.hstack((result, data))
    return result


def pixel_rounder(n, mode):
    if mode==True or mode=='round':
        return round(n)
    elif mode=='ceil':
        return math.ceil(n)
    elif mode=='floor':
        return math.floor(n)
    else:
        return n


def pixel_ij(x, rounding=True):
    if isinstance(x, np.ndarray):
        x = x.tolist()
    return tuple(pixel_rounder(i, rounding) for i in (
        x if isinstance(x, tuple) or isinstance(x, list) else (x,x)
    ))


def ucolors(num_colors):
    # uniform color generator
    colors=[]
    for i in np.arange(0., 360., 360. / num_colors):
        hue = i/360.
        lightness = (50 + np.random.rand() * 10)/100.
        saturation = (90 + np.random.rand() * 10)/100.
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    return colors

coco_parts_dict = {
    "ear_right-eye_right": (4, 2),
    "eye_right-nose": (2, 0),
    "nose-eye_left": (0, 1),
    "eye_left-ear_left": (1, 3),

    "wrist_right-elbow_right": (10, 8),
    "elbow_right-shoulder_right": (8, 6),
    "shoulder_right-shoulder_left": (6, 5),
    "shoulder_left-elbow_left": (5, 7),
    "elbow_left-wrist_left": (7, 9),

    "ankle_right-knee_right": (16, 14),
    "knee_right-hip_right": (14, 12),
    "hip_right-hip_left": (12, 11),
    "hip_left-knee_left": (11, 13),
    "knee_left-ankle_left": (13, 15)
}

coco_parts = [
    (4, 2),  #  0, ear_right - eye_right
    ( 2, 0),  #  1, eye_right - nose
    ( 0, 1),  #  2, nose - eye_left
    ( 1, 3),  #  3, eye_left - ear_left

    (10, 8),  #  4, wrist_right - elbow_right
    ( 8, 6),  #  5, elbow_right - shoulder_right
    ( 6, 5),  #  6, shoulder_right - shoulder_left
    ( 5, 7),  #  7, shoulder_left - elbow_left
    ( 7, 9),  #  8, elbow_left - wrist_left

    (16,14),  #  9, ankle_right - knee_right
    (14,12),  # 10, knee_right - hip_right
    (12,11),  # 11, hip_right - hip_left
    (11,13),  # 12, hip_left - knee_left
    (13,15),  # 13, knee_left- ankle_left
]
coco_parts_ext = coco_parts + [
    ( 0,17),  # 14, nose - nose_root
    ( 9,19),  # 15, wrist_left - thumb_left
    (10,20),  # 16, wrist_right - thumb_right
    (15,21),  # 17, ankle_left - toe_left
    (16,22),  # 18, ankle_right - toe_right
]

coco_part_colors = ucolors(len(coco_parts))
coco_part_colors_ext = coco_part_colors + ucolors(len(coco_parts_ext)-len(coco_parts))

coco_keypoints = [
    'nose',             #  0
    'eye_left',         #  1
    'eye_right',        #  2
    'ear_left',         #  3
    'ear_right',        #  4

    'shoulder_left',    #  5
    'shoulder_right',   #  6
    'elbow_left',       #  7
    'elbow_right',      #  8
    'wrist_left',       #  9
    'wrist_right',      # 10

    'hip_left',         # 11
    'hip_right',        # 12
    'knee_left',        # 13
    'knee_right',       # 14
    'ankle_left',       # 15
    'ankle_right',      # 16
]

coco_keypoints_ext = coco_keypoints + [
    'nose_root',    # 17
    'body_upper',   # 18
    'thumb_left',   # 19
    'thumb_right',  # 20
    'toe_left',     # 21
    'toe_right',    # 22
]

def c255(c):
    # color format utility
    if c is None:
        return None
    if isinstance(c, str):
        c = {
            'r': (1,0,0),
            'g': (0,1,0),
            'b': (0,0,1),
            'k': 0,
            'w': 1,
            't': (0,1,1),
            'm': (1,0,1),
            'y': (1,1,0),
            'a': (0,0,0,0),
        }[c]
    if isinstance(c, list) or isinstance(c, tuple):
        if len(c)==3:
            c = c + (1,)
        elif len(c)==1:
            c = (c,c,c,1)
        c = tuple(int(255*q) for q in c)
    else:
        c = int(255*c)
        c = (c,c,c,255)
    return c

def draw_rect(image, corner, size, w=1, c='r', f=None):
    corner = pixel_ij(corner, rounding=True)
    size = pixel_ij(size, rounding=True)
    w = max(1, round(w))
    c = c255(c)
    f = c255(f)
    ans = image.copy()
    d = ImageDraw.Draw(ans)
    d.rectangle(
        [corner[1], corner[0], corner[1]+size[1]-1, corner[0]+size[0]-1],
        fill=f, outline=c, width=w,
    )
    return ans

def draw_dot(d: ImageDraw.ImageDraw, point, s=1, c='r'):
    c = c255(c)
    x,y = pixel_ij(point, rounding=False)
    d.ellipse(
        [(y-s,x-s), (y+s,x+s)],
        fill=c,
    )

def draw_line(d: ImageDraw.ImageDraw, a, b, w=1, c='r'):
    a = pixel_ij(a, rounding=False)
    b = pixel_ij(b, rounding=False)
    c = c255(c)
    w = max(1, round(w))
    d.line([a[::-1], (b[1]-1,b[0]-1)], fill=c, width=w)

    

def visualize_pos_keypoints(image: Image.Image, bbox=None, keypoints=None, scores=None):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if bbox is not None:
        image = draw_rect(image, *bbox, c='r', w=2)
    image_rst = image.copy()
    d = ImageDraw.Draw(image_rst)
    if keypoints is not None:
        if isinstance(keypoints, dict):
            keypoints = np.asarray([keypoints[k] for k in coco_keypoints])
        for (a,b),c in zip(coco_parts, coco_part_colors):
            draw_line(d, keypoints[a], keypoints[b], w=5, c=c)
        keypoints = keypoints[:len(coco_keypoints)]
        for ii, kp in enumerate(keypoints):
            draw_dot(d, kp, s=5, c='r')
        font = ImageFont.truetype("assets/arial.ttf", size=20)
        if scores is not None:
            # fnt = ImageFont.load_default(size=30)
            for ii, kp in enumerate(keypoints):
                s = scores[ii]
                d.text((kp[1], kp[0]), f'{int(round(s * 100))}', font=font, fill=(0, 0, 0, 255), stroke_fill=(255, 255, 255, 255), stroke_width=2)

    return image_rst



def pil_draw_text(image: Image.Image, text, point, fill=(0, 0, 0, 255), font_size=20, stroke_fill=(255, 255, 255, 255), font=None, stroke_width=0):
    is_ndarray = False
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
        is_ndarray = True
    d = ImageDraw.Draw(image)
    if font is None:
        font = ImageFont.truetype("assets/arial.ttf", size=font_size)
    if not isinstance(text, list):
        text = [text]
        point = [point]
    for t, p in zip(text, point):
        d.text(p, t, font=font, fill=fill, stroke_fill=stroke_fill, stroke_width=stroke_width)
    if is_ndarray:
        image = np.array(image)
    return image




# https://github.com/hysts/anime-face-detector/blob/main/assets/landmarks.jpg
FACE_BOTTOM_OUTLINE = np.arange(0, 5)
LEFT_EYEBROW = np.arange(5, 8)
RIGHT_EYEBROW = np.arange(8, 11)
LEFT_EYE_TOP = np.arange(11, 14)
LEFT_EYE_BOTTOM = np.arange(14, 17)
RIGHT_EYE_TOP = np.arange(17, 20)
RIGHT_EYE_BOTTOM = np.arange(20, 23)
NOSE = np.array([23])
MOUTH_OUTLINE = np.arange(24, 28)

FACE_OUTLINE_LIST = [FACE_BOTTOM_OUTLINE, LEFT_EYEBROW, RIGHT_EYEBROW]
LEFT_EYE_LIST = [LEFT_EYE_TOP, LEFT_EYE_BOTTOM]
RIGHT_EYE_LIST = [RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM]
NOSE_LIST = [NOSE]
MOUTH_OUTLINE_LIST = [MOUTH_OUTLINE]

# (indices, BGR color, is_closed)
FACE_CONTOURS = [
    (FACE_OUTLINE_LIST, (0, 170, 255), False),
    (LEFT_EYE_LIST, (50, 220, 255), False),
    (RIGHT_EYE_LIST, (50, 220, 255), False),
    (NOSE_LIST, (255, 30, 30), False),
    (MOUTH_OUTLINE_LIST, (255, 30, 30), True),
]

def visualize_box(image,
                  box,
                  score,
                  lt,
                  box_color=(0, 255, 0),
                  text_color=(255, 255, 255),
                  show_box_score=True):
    cv2.rectangle(image, tuple(box[:2]), tuple(box[2:]), box_color, lt)
    if not show_box_score:
        return
    cv2.putText(image,
                f'{round(score * 100, 2)}%', (box[0], box[1] - 2),
                0,
                lt / 2,
                text_color,
                thickness=max(lt, 1),
                lineType=cv2.LINE_AA)


def visualize_landmarks(image, pts, lt, landmark_score_threshold):
    for *pt, score in pts:
        pt = tuple(np.round(pt).astype(int))
        if score < landmark_score_threshold:
            color = (0, 255, 255)
        else:
            color = (0, 0, 255)
        cv2.circle(image, pt, lt, color, cv2.FILLED)


def draw_polyline(image, pts, color, closed, lt, skip_contour_with_low_score,
                  score_threshold):
    if skip_contour_with_low_score and (pts[:, 2] < score_threshold).any():
        return
    pts = np.round(pts[:, :2]).astype(int)
    cv2.polylines(image, np.array([pts], dtype=np.int32), closed, color, lt)


def visualize_face_contour(image, pts, lt, skip_contour_with_low_score,
                      score_threshold):
    for indices_list, color, closed in FACE_CONTOURS:
        for indices in indices_list:
            draw_polyline(image, pts[indices], color, closed, lt,
                          skip_contour_with_low_score, score_threshold)



def visualize_facedet_output(image: np.ndarray,
              preds: np.ndarray,
              face_score_threshold: float = 0.5,
              landmark_score_threshold: float = 0.3,
              show_box_score: bool = True,
              draw_contour: bool = True,
              skip_contour_with_low_score=False):
    res = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for pred in preds:
        box = pred['bbox']
        box, score = box[:4], box[4]
        box = np.round(box).astype(int)
        pred_pts = pred['keypoints']

        # line_thickness
        lt = max(2, int(3 * (box[2:] - box[:2]).max() / 256))

        visualize_box(res, box, score, lt, show_box_score=show_box_score)
        if draw_contour:
            visualize_face_contour(
                res,
                pred_pts,
                lt,
                skip_contour_with_low_score=skip_contour_with_low_score,
                score_threshold=landmark_score_threshold)
        visualize_landmarks(res, pred_pts, lt, landmark_score_threshold)

    res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
    return res



FACE_LABEL2NAME = {
    0:'background', 
    1:'skin', # it's face
    2:'l_brow', 3:'r_brow',
    4:'l_eye', 5:'r_eye', 
    6:'eye_g',  # glass 
    7:'l_ear', 8:'r_ear', 
    9:'ear_r',  # not used
    10:'nose', 11:'mouth',
    12:'u_lip', 13:'l_lip', # both not used 
    14:'neck', 
    15:'neck_l',    # not used
    16:'cloth',     # mostly collar
    17:'hair', 
    18:'hat'
}

VALID_FACE_GROUPS = {
    'face': [1],
    'brows': [2, 3],
    'eyes': [4, 5],
    'glass': [6],
    'ears': [7, 8],
    'nose': [10],
    'mouth': [11],
    'neck': [14],
    'cloth': [16],
    'hair': [17],
    'hat': [18]
}


def visualize_segs(labels: Union[np.ndarray, List], src_img: np.ndarray = None, image_weight = 0.5, colors=None, label2name_map=None, output_dtype='numpy'):
    '''
    labels: (H, W) components map or (N, H, W) masks
    '''

    if label2name_map is None:
        label2name_map = {l: str(l) for l in range(len(labels))}

    if isinstance(labels, list):
        labels = [label[None] if label.ndim == 2 else label for label in labels]
        labels = np.concatenate(labels)

    # if colors is None:
    #     colors = np.array([get_color(ii) for ii in range(len(label2name_map))], dtype=np.float32) / 255.
    
    # h, w = labels.shape[-2:]
    # vis = np.zeros((h, w, 3), dtype=np.float32)

    # if labels.ndim == 3:
    #     assert labels.shape[0] == len(label2name_map)
    # else:
    #     assert labels.ndim == 2

    # for label in label2name_map:
    #     if labels.ndim == 2:
    #         mask = (labels == label).astype(np.float32)[..., None]
    #     else:
    #         mask = labels[label].astype(np.float32)[..., None]
    #     vis += mask * colors[label][:3].reshape((1, 1, -1))

    # if src_img is not None:
    #     if np.max(src_img) > 1:
    #         src_img = src_img.astype(np.float32) / 255.
    #     vis = image_weight * src_img + (1 - image_weight) * vis

    n_components = len(labels)
    colors = np.array([get_color(ii) for ii in range(n_components)], dtype=np.uint8)

    img = src_img
    mask_list = labels
    colored_list = [src_img]
    mask_full = np.zeros_like(mask_list[0]).astype(np.float32)
    c = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    for ii, mask in enumerate(mask_list):
        color = np.array(colors[ii][:3])
        mask_f = mask.astype(np.float32)
        mask_full += mask_f
        # c = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
        c[np.where(mask)] = (color).astype(np.uint8)
        # c[..., -1] = np.clip(mask.astype(np.float32) * 255 * (1 - image_weight), 0, 255).astype(np.uint8)
        # colored_list.append(c)

    # colored_final = img_alpha_blending(colored_list)
    mask_full = np.clip(mask_full[..., None], 0, 1) * (1-image_weight)
    vis = np.round(c * mask_full + img * (1-mask_full)).astype(np.uint8)

    # vis = (np.clip(vis, 0, 1) * 255).astype(np.uint8)
    if output_dtype.lower() == 'pil':
        return Image.fromarray(vis)
    return vis


COLOR_PALETTE = [
    (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
    (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
    (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30),
    (165, 42, 42), (255, 77, 255), (0, 226, 252), (182, 182, 255),
    (0, 82, 0), (120, 166, 157), (110, 76, 0), (174, 57, 255),
    (199, 100, 0), (72, 0, 118), (255, 179, 240), (0, 125, 92),
    (209, 0, 151), (188, 208, 182), (0, 220, 176), (255, 99, 164),
    (92, 0, 73), (133, 129, 255), (78, 180, 255), (0, 228, 0),
    (174, 255, 243), (45, 89, 255), (134, 134, 103), (145, 148, 174),
    (255, 208, 186), (197, 226, 255), (171, 134, 1), (109, 63, 54),
    (207, 138, 255), (151, 0, 95), (9, 80, 61), (84, 105, 51),
    (74, 65, 105), (166, 196, 102), (208, 195, 210), (255, 109, 65),
    (0, 143, 149), (179, 0, 194), (209, 99, 106), (5, 121, 0),
    (227, 255, 205), (147, 186, 208), (153, 69, 1), (3, 95, 161),
    (163, 255, 0), (119, 0, 170), (0, 182, 199), (0, 165, 120),
    (183, 130, 88), (95, 32, 0), (130, 114, 135), (110, 129, 133),
    (166, 74, 118), (219, 142, 185), (79, 210, 114), (178, 90, 62),
    (65, 70, 15), (127, 167, 115), (59, 105, 106), (142, 108, 45),
    (196, 172, 0), (95, 54, 80), (128, 76, 255), (201, 57, 1),
    (246, 0, 122), (191, 162, 208), (255, 255, 128), (147, 211, 203),
    (150, 100, 100), (168, 171, 172), (146, 112, 198), (210, 170, 100),
    (92, 136, 89), (218, 88, 184), (241, 129, 0), (217, 17, 255),
    (124, 74, 181), (70, 70, 70), (255, 228, 255), (154, 208, 0),
    (193, 0, 92), (76, 91, 113), (255, 180, 195), (106, 154, 176),
    (230, 150, 140), (60, 143, 255), (128, 64, 128), (92, 82, 55),
    (254, 212, 124), (73, 77, 174), (255, 160, 98), (255, 255, 255),
    (104, 84, 109), (169, 164, 131), (225, 199, 255), (137, 54, 74),
    (135, 158, 223), (7, 246, 231), (107, 255, 200), (58, 41, 149),
    (183, 121, 142), (255, 73, 97), (107, 142, 35), (190, 153, 153),
    (146, 139, 141), (70, 130, 180), (134, 199, 156), (209, 226, 140),
    (96, 36, 108), (96, 96, 96), (64, 170, 64), (152, 251, 152),
    (208, 229, 228), (206, 186, 171), (152, 161, 64), (116, 112, 0),
    (0, 114, 143), (102, 102, 156), (250, 141, 255)
]

class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        # hexs = ('FF1010', '10FF10', 'FFF010', '100FFF', 'c0c0c0', 'FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
        #         '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        hexs = [
            '#4363d8',
            '#9A6324',
            '#808000',
            '#469990',
            '#000075',
            '#e6194B',
            '#f58231',
            '#ffe119',
            '#bfef45',
            '#3cb44b',
            '#42d4f4',
            '#800000',
            '#911eb4',
            '#f032e6',
            '#fabed4',
            '#ffd8b1',
            '#fffac8',
            '#aaffc3',
            '#dcbeff',
            '#a9a9a9',
            '#006400',
            '#4169E1',
            '#8B4513',
            '#FA8072',
            '#87CEEB',
            '#FFD700',
            '#ffffff',
            '#000000',
        ]
        self.palette = [self.hex2rgb(f'#{c}') if not c.startswith('#') else self.hex2rgb(c) for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))
    
DEFAULT_COLOR_PALETTE = Colors()
def get_color(idx):
    if idx == -1:
        return 255
    else:
        return DEFAULT_COLOR_PALETTE(idx)



def vis_parts(srcd: str, tag_list, nmax_samples=12, cols=4):
    partsd = osp.join(srcd, 'parts')
    rst_list = []
    nparts = 0
    for tag in tag_list:
        p = osp.join(partsd, tag + '_vis.png')
        if not osp.exists(p):
            continue
        img = Image.open(p)
        pil_draw_text(img, tag, point=(0, 0), font_size=128, stroke_width=12)
        rst_list.append(img)
        if len(rst_list) >= nmax_samples:
            vis = imglist2imgrid(rst_list, cols=cols)
            Image.fromarray(vis).save(osp.join(srcd, f'part_vis{nparts}.jpg'), q=97)
            rst_list = []
            nparts += 1

    if len(rst_list) > 0:
        vis = imglist2imgrid(rst_list, cols=cols)
        Image.fromarray(vis).save(osp.join(srcd, f'part_vis{nparts}.jpg'), q=97)


def imglist2imgrid_with_tags(
        img_list, tag_list, cols=4, output_type='numpy', skip_empty=False, fix_size=None, tag_breakn=4, font_size=-1, stroke_width=-1, font_pos = (0, 0)):
    
    def _wrap_text(tags):
        '''
        maybe wrap according to actual render size
        '''
        if isinstance(tags, str):
            tags = tags.split(',')
        caption = ''
        nadded = 0
        for t in tags:
            if nadded >= tag_breakn:
                nadded = 1
                caption = caption + '\n' + t
            else:
                nadded += 1
                if caption == '':
                    caption = t
                else:
                    caption = caption + ',' + t
        return caption

    new_img_lst = []
    for ii, img in enumerate(img_list):
        if skip_empty and tag_list[ii].strip() == '':
            continue

        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        else:
            assert isinstance(img, Image.Image)
            img = img.copy()
        ih, iw = img.height, img.width
        if font_size < 0:
            font_size = max(ih, iw) // 16
        if stroke_width < 0:
            stroke_width = font_size // 10

        pil_draw_text(img, _wrap_text(tag_list[ii]), point=font_pos, font_size=font_size, stroke_width=stroke_width)
        new_img_lst.append(img)
        
    return imglist2imgrid(new_img_lst, cols=cols, output_type=output_type, fix_size=fix_size)



class JointType(IntEnum):
    Nose = 0
    Neck = 1
    RightShoulder = 2
    RightElbow = 3
    RightHand = 4
    LeftShoulder = 5
    LeftElbow = 6
    LeftHand = 7
    RightWaist = 8
    RightKnee = 9
    RightFoot = 10
    LeftWaist = 11
    LeftKnee = 12
    LeftFoot = 13
    RightEye = 14
    LeftEye = 15
    RightEar = 16
    LeftEar = 17
    RightToes = 18
    LeftToes = 19
    RightFist = 20
    LeftFist = 21
    Spine = 22


def plot_faceparsing(img: np.ndarray, pose, face_landmarks):

    landmark_params = {
        'moe_joint_indices': {
            'nose': JointType.Nose,
            'L_eye': JointType.LeftEye,
            'R_eye': JointType.RightEye,
            'L_ear': JointType.LeftEar,
            'R_ear': JointType.RightEar,
            'neck': JointType.Neck,
            'L_shoulder': JointType.LeftShoulder,
            'R_shoulder': JointType.RightShoulder,
            'L_elbow': JointType.LeftElbow,
            'R_elbow': JointType.RightElbow,
            'L_hand': JointType.LeftHand,
            'R_hand': JointType.RightHand,
            'L_waist': JointType.LeftWaist,
            'R_waist': JointType.RightWaist,
            'L_knee': JointType.LeftKnee,
            'R_knee': JointType.RightKnee,
            'L_foot': JointType.LeftFoot,
            'R_foot': JointType.RightFoot
        },
        'limbs_point_plot': [
            ['neck', 'R_waist'],
            ['R_waist', 'R_knee'],
            ['R_knee', 'R_foot'],
            ['neck', 'L_waist'],
            ['L_waist', 'L_knee'],
            ['L_knee', 'L_foot'],
            ['neck', 'R_shoulder'],
            ['R_shoulder', 'R_elbow'],
            ['R_elbow', 'R_hand'],
            ['R_shoulder', 'R_ear'],
            ['neck', 'L_shoulder'],
            ['L_shoulder', 'L_elbow'],
            ['L_elbow', 'L_hand'],
            ['L_shoulder', 'L_ear'],
            ['neck', 'nose'],
            ['nose', 'R_eye'],
            ['nose', 'L_eye'],
            ['R_eye', 'R_ear'],
            ['L_eye', 'L_ear']
        ]
    }

    landmark_all_groups = ['outline',
                        'nose',
                        'mouth',
                        'left_eye_brow',
                        'right_eye_brow',
                        'left_eye_outline',
                        'right_eye_outline',
                        'right_eye',
                        'left_eye']

    def _plot_pose(orig_img, pose, line_width=2, circle_size=3):
        limb_colors = [
            [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255],
            [0, 85, 255], [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0.],
            [255, 0, 85], [170, 255, 0], [85, 255, 0], [170, 0, 255.], [0, 0, 255],
            [0, 0, 255], [255, 0, 255], [170, 0, 255], [255, 0, 170],
        ]
        joint_colors = [
            [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0],
            [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
            [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255],
            [255, 0, 255], [255, 0, 170], [255, 0, 85]]

        canvas = orig_img

        for i, (limb, color) in enumerate(zip(landmark_params['limbs_point_plot'], limb_colors)):
            if i != 9 and i != 13:  # don't show ear-shoulder connection
                keya = limb[0]
                keyb = limb[1]
                xa = pose['pose'][keya]['x']
                ya = pose['pose'][keya]['y']
                va = pose['pose'][keya]['v']
                xb = pose['pose'][keyb]['x']
                yb = pose['pose'][keyb]['y']
                vb = pose['pose'][keyb]['v']
                if va > 0 and vb > 0:
                    cv2.line(canvas, (xa, ya), (xb, yb), color, line_width)

        for key, color in zip(list(landmark_params['moe_joint_indices'].keys()), joint_colors):
            x = pose['pose'][key]['x']
            y = pose['pose'][key]['y']
            v = pose['pose'][key]['v']
            if v != 0:
                cv2.circle(canvas, (x, y), circle_size, color, -1)
        return canvas

    def _plot_face(orig_img, landmarks, draw_line=True, line_width=2, circle_size=3):
        
        group_line_color = {
            'outline': [0, 255, 0],
            'left_eye_brow': [0, 255, 85], 
            'right_eye_brow': [0, 255, 170], 
            'left_eye_outline': [0, 255, 255], 
            'right_eye_outline': [0, 170, 255],
            'nose': [0, 85, 255],
            'mouth': [255, 85, 0]
        }
        
        group_colors = {
            'nose': [0, 0, 0],
            'mouth': [127, 127, 127],
            'left_eye': [0, 127, 0],
            'right_eye': [0, 0, 127],
            'outline': [255, 0, 0],
            'left_eye_outline': [255, 0, 255],
            'right_eye_outline': [255, 255, 0],
            'left_eye_brow': [0, 255, 0],
            'right_eye_brow': [0, 0, 255],
        }

        group_line = {
            'outline': [[0, 1],
                        [1, 2],
                        [2, 3],
                        [3, 4],
                        [4, 5],
                        [5, 6],
                        [6, 7],
                        [7, 8]],
            'left_eye_brow': [[0, 1], [1, 2]],
            'right_eye_brow': [[0, 1], [1, 2]],
            'left_eye_outline': [[0, 1],
                                [1, 2],
                                [2, 3],
                                [3, 4],
                                [4, 5],
                                [5, 0]],
            'right_eye_outline': [[0, 1],
                                [1, 2],
                                [2, 3],
                                [3, 4],
                                [4, 5],
                                [5, 0]],
            'nose': [[0, 1]],
            'mouth': [[0, 1],
                    [1, 2],
                    [2, 3],
                    [3, 4],
                    [4, 5],
                    [5, 0]],
        }

        canvas = orig_img
        if draw_line:
            for k, color in group_line_color.items():
                if k in landmarks:
                    for limb in group_line[k]:
                        keya = limb[0]
                        keyb = limb[1]

                        xa = int(landmarks[k][keya][0])
                        ya = int(landmarks[k][keya][1])
                        xb = int(landmarks[k][keyb][0])
                        yb = int(landmarks[k][keyb][1])

                        cv2.line(canvas, (xa, ya), (xb, yb), color, line_width)

        for k, color in group_colors.items():
            if k in landmarks:
                for points in landmarks[k]:
                    x = int(points[0])
                    y = int(points[1])
                    cv2.circle(canvas, (x, y), circle_size, color, -1)
        return canvas

    def shift_position(pos, left, top):
        new_pos = pos.copy()
        new_pos[:, 0] += left
        new_pos[:, 1] += top
        return new_pos

    canvas = img.copy()
    _plot_pose(canvas, pose)

    def get_points(group_names=[]):
        if isinstance(group_names, list):
            ret = {}
            for g in group_names:
                ret[g] = shift_position(face_landmarks['points'][g], face_landmarks['left'], face_landmarks['top'])
            return ret

        return shift_position(face_landmarks['points'][group_names], face_landmarks['left'], face_landmarks['top'])

    pos = np.array(face_landmarks['points'])
    
    _plot_face(canvas, get_points(landmark_all_groups))
    return canvas


RECT_LINES = ((0, 1), (1, 2), (2, 3), (3, 0))

def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])


def labelcolormap(N):
    if N == 35:  # cityscape
        cmap = np.array([(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (111, 74, 0), (81, 0, 81),
                         (128, 64, 128), (244, 35, 232), (250, 170, 160), (230, 150, 140), (70, 70, 70), (102, 102, 156), (190, 153, 153),
                         (180, 165, 180), (150, 100, 100), (150, 120, 90), (153, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0),
                         (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
                         (0, 60, 100), (0, 0, 90), (0, 0, 110), (0, 80, 100), (0, 0, 230), (119, 11, 32), (0, 0, 142)],
                        dtype=np.uint8)
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i + 1  # let's give 0 a color
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7 - j))
                g = g ^ (np.uint8(str_id[-2]) << (7 - j))
                b = b ^ (np.uint8(str_id[-3]) << (7 - j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b

    return cmap
    

def plot_points_lines(orig_img, points, lines=RECT_LINES, line_width=2, circle_size=3, ref_pos=(0, 0)):
    colors_p = labelcolormap(len(points))
    colors_l = labelcolormap(len(lines))

    c = orig_img.shape[-1]
    
    canvas = orig_img.copy()
    for i in range(len(lines)):
        keya = lines[i][0]
        keyb = lines[i][1]
        xa = int(points[keya][0] + ref_pos[0])
        ya = int(points[keya][1] + ref_pos[1])
        xb = int(points[keyb][0] + ref_pos[0])
        yb = int(points[keyb][1] + ref_pos[1])
        # print(colors_l[i])
        color = [int(colors_l[i][0]), int(colors_l[i][1]), int(colors_l[i][2])]
        cv2.line(canvas, (xa, ya), (xb, yb), color, line_width)
        
    for i in range(len(points)):
        x = int(points[i][0] + ref_pos[0])
        y = int(points[i][1] + ref_pos[1])
        color = [int(colors_p[i][0]), int(colors_p[i][1]), int(colors_p[i][2])]
        if c == 4:
            color.append(255)
        cv2.circle(canvas, (x, y), circle_size, color, -1)
    return canvas