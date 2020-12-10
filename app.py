import datetime
import os
from types import CodeType

import cv2
import numpy as np

from helpers import *

MIN_MATCHES = 20

TEST_VIDEO = 'wale_tehouse_720p.mp4'
RESULT_NAME = 'wale_tehouse_720p_ORB'

references = [['wale.jpeg', 'dolphin/dolphin.obj', 2],
              ['teahouse.jpeg', 'tea/tea.obj', 1]]

DRAW_MATCHES = False
COLOR_USE = True


font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (20, 40)
fontScale = 0.5
fontColor = (0, 0, 0)
lineType = 1


def main():
    def load_reference_object(reference_image_path, object_path, col):

        reference_image = cv2.imread(os.path.join(
            dir_name, f'reference/{reference_image_path}'), 0)

        kp_model, des_model = orb.detectAndCompute(reference_image, None)

        obj = OBJ(os.path.join(dir_name, f'models/{object_path}'), swapyz=True)
        return kp_model, des_model, obj, reference_image, col

    def render_object(loaded_reference_object, frame, homography, changed_frame, obj_name):
        kp_model, des_model, obj, reference_image, col = loaded_reference_object
        kp_frame, des_frame = orb.detectAndCompute(frame, None)
        matches = bf.match(des_model, des_frame)
        matches = sorted(matches, key=lambda x: x.distance)[:]

        if len(matches) > MIN_MATCHES:
            src_pts = np.float32(
                [kp_model[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            homography, mask = cv2.findHomography(
                src_pts, dst_pts, cv2.RANSAC, 5.0)

            if homography is not None:
                try:
                    projection = projection_matrix(
                        camera_parameters, homography)
                    print(f'{obj_name} - ',
                          np.mean([x.distance for x in matches[:10]]))
                    if np.mean([x.distance for x in matches[:10]]) < 40:
                        changed_frame = render(
                            changed_frame, obj, projection, reference_image, COLOR_USE, col)
                except:
                    CodeType
            if DRAW_MATCHES:
                changed_frame = cv2.drawMatches(
                    reference_image, kp_model, changed_frame, kp_frame, matches[:10], 0, flags=2)

            return changed_frame

        else:
            print("Not enough matches found - %d/%d" %
                  (len(matches), MIN_MATCHES))
            return frame

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(f'{RESULT_NAME}.avi', fourcc, 60.0, (1280, 720))

    homography = None

    camera_parameters = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])

    orb = cv2.ORB_create(nfeatures=500, scoreType=cv2.ORB_HARRIS_SCORE)
    # orb = cv2.BRISK_create(thresh=40)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    dir_name = os.getcwd()

    references_loaded = [load_reference_object(*ref) for ref in references]

    cap = cv2.VideoCapture(f'test_videos/{TEST_VIDEO}')

    fps_counted = 0
    while (cap.isOpened()):
        start_date_time = datetime.datetime.now()

        ret, frame = cap.read()
        if not ret:
            print("Unable to capture video")
            return

        changed_frame = frame.copy()
        for ref_idx, loaded_reference_object in enumerate(references_loaded):
            changed_frame = render_object(
                loaded_reference_object, frame, homography, changed_frame, references[ref_idx][0])

        cv2.putText(changed_frame, f'FPS: {10**6/(datetime.datetime.now() - start_date_time).microseconds}',
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)

        cv2.imshow('frame', changed_frame)
        datetime.timedelta(seconds=1)
        print(changed_frame.shape)
        if ret == True:
            writer.write(changed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    writer.release()

    cv2.destroyAllWindows()
    return 0


if __name__ == '__main__':
    main()
