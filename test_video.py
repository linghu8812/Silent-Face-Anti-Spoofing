# import os
import argparse
import insightface
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
import cv2
import numpy as np
# import matplotlib.pyplot as plt


# Plotting functions ---------------------------------------------------------------------------------------------------
def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * max(img.shape[0:2])) + 1  # line thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='face anti spoofing proto code')
    parser.add_argument('video', help='video file name.')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id.')
    parser.add_argument('--rotate', action='store_true', help='rotate frame.')
    args = parser.parse_args()

    # RetinaFace Detector
    detector = insightface.model_zoo.get_model(
        'retinaface_mnet025_v2')  # can replace with your own face detector
    # self.detector = insightface.model_zoo.get_model('retinaface_r50_v1')
    detector.prepare(ctx_id=args.gpu)

    # Anti Spoof Classifier
    model_test = AntiSpoofPredict(args.gpu)
    image_cropper = CropImage()
    h_input, w_input, model_type, scale = parse_model_name('2.7_80x80_MiniFASNetV2.pth')

    # Start pipeline
    video_capture = cv2.VideoCapture(args.video)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = 'MJPG'
    size = (height, width) if args.rotate else (width, height)
    video_writer = cv2.VideoWriter(args.video.replace('mp4', 'avi').replace('mov', 'avi'),
                                   cv2.VideoWriter_fourcc(*fourcc), fps, size)
    while True:
        success, frame = video_capture.read()
        if not success:
            video_capture.release()
            video_writer.release()
            break
        if args.rotate:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        bboxes, _ = detector.detect(frame, threshold=0.8)
        for bbox in bboxes:
            left, top, right, bottom, _ = bbox.astype(np.int32)
            if right - left < 50 or bottom - top < 50:
                continue
            param = {
                "org_img": frame,
                "bbox": bbox,
                "scale": scale,
                "out_w": w_input,
                "out_h": h_input,
                "crop": True,
            }
            if scale is None:
                param["crop"] = False
            roi = image_cropper.crop(**param)
            # plt.imshow(roi)
            # plt.show()
            prediction = model_test.predict(roi, './resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth')
            result = np.argmax(prediction)
            value = prediction[0][result]
            if result == 1:
                plot_one_box(bbox, frame, (0, 255, 0), f'real{value:.2f}')
            else:
                plot_one_box(bbox, frame, (0, 0, 255), f'fake{value:.2f}')
        video_writer.write(frame)
