
# coding: utf-8
#!/usr/bin/env pypy

"""

Description::

Facial Recognition Model

"""

import cv2
import numpy as np
import os
import sys


def set_trace():
    """A Poor mans break point"""
    # without this in iPython debugger can generate strange characters.
    from IPython.core.debugger import Pdb
    Pdb().set_trace(sys._getframe().f_back)


class FaceDetector:
    def __init__(
        self, face_casc='haarcascade_frontalface_default.xml',
        left_eye_casc='haarcascade_lefteye_2splits.xml',
        right_eye_casc='haarcascade_righteye_2splits.xml', scale_factor=4):

        self.scale_factor = scale_factor

        self.face_casc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        if self.face_casc.empty():
            print 'Warning: Could not load face cascade:', 'haarcascade_frontalface_default.xml'
            raise SystemExit
        self.left_eye_casc = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')
        if self.left_eye_casc.empty():
            print 'Warning: Could not load left eye cascade:', 'haarcascade_lefteye_2splits.xml'
            raise SystemExit
        self.right_eye_casc = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')
        if self.right_eye_casc.empty():
            print 'Warning: Could not load right eye cascade:', 'haarcascade_righteye_2splits.xml'
            raise SystemExit

    def detect(self, frame):
        frameCasc = cv2.cvtColor(cv2.resize(frame,
                                            (0, 0),
                                            fx=1.0 / self.scale_factor,
                                            fy=1.0 / self.scale_factor),
                                 cv2.COLOR_RGB2GRAY)
        faces = self.face_casc.detectMultiScale(frameCasc,
                                                scaleFactor=1.1,
                                                minNeighbors=3,
                                                flags=cv2.CASCADE_FIND_BIGGEST_OBJECT) * self.scale_factor

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 255, 0), 2)
            head = cv2.cvtColor(frame[y:y + h, x:x + w],
                                cv2.COLOR_RGB2GRAY)
            return True, frame, head
        return False, frame, None

    def align_head(self, head):
        """Aligns a head region using affine transformations
            This method preprocesses an extracted head region by rotating
            and scaling it so that the face appears centered and up-right.
            The method returns True on success (else False) and the aligned
            head region (head). Possible reasons for failure are that one or
            both eye detectors fail, maybe due to poor lighting conditions.
            :param head: extracted head region
            :returns: success, head
        """
        
        # -----> PUT IN SOME CODE TO DEAL WITH NONETYPES FOR head below


        height, width = head.shape[:2]

        # detect left eye
        left_eye_region = head[int(0.2 * height):int(0.8 * height), int(0.2 * width):int(0.5 * width)]
        left_eye = self.left_eye_casc.detectMultiScale(
            left_eye_region,
            scaleFactor=1.1,
            minNeighbors=3,
            flags=cv2.CASCADE_FIND_BIGGEST_OBJECT)
        left_eye_center = None
        for (xl, yl, wl, hl) in left_eye:
            # find the center of the detected eye region
            left_eye_center = np.array([0.1 * width + xl + wl / 2,
                                        0.2 * height + yl + hl / 2])
            break  # need only look at first, largest eye
        # detect right eye
        right_eye_region = head[int(0.2 * height):int(0.8 * height), int(0.5 * width):int(0.9 * width)]
        right_eye = self.right_eye_casc.detectMultiScale(
            right_eye_region,
            scaleFactor=1.1,
            minNeighbors=3,
            flags=cv2.CASCADE_FIND_BIGGEST_OBJECT)
        right_eye_center = None
        for (xr, yr, wr, hr) in right_eye:
            # find the center of the detected eye region
            right_eye_center = np.array([0.5 * width + xr + wr / 2,
                                         0.2 * height + yr + hr / 2])
            break  # need only look at first, largest eye
        # need both eyes in order to align face
        # else break here and report failure (False)
        if left_eye_center is None or right_eye_center is None:
            return False, head

        # we want the eye to be at 25% of the width, and 20% of the height
        # resulting image should be square (desired_img_width,
        # desired_img_height)
        desired_eye_x = 0.25
        desired_eye_y = 0.2
        desired_img_width = 200
        desired_img_height = desired_img_width

        # get center point between the two eyes and calculate angle
        eye_center = (left_eye_center + right_eye_center) / 2
        eye_angle_deg = np.arctan2(right_eye_center[1] - left_eye_center[1],
                                   right_eye_center[0] - left_eye_center[0]) \
            * 180.0 / 3.1415927

        # scale distance between eyes to desired length
        eyeSizeScale = (1.0 - desired_eye_x * 2) * desired_img_width / \
            np.linalg.norm(right_eye_center - left_eye_center)

        # get rotation matrix
        rot_mat = cv2.getRotationMatrix2D(tuple(eye_center), eye_angle_deg,
                                          eyeSizeScale)

        # shift center of the eyes to be centered in the image
        rot_mat[0, 2] += desired_img_width * 0.5 - eye_center[0]
        rot_mat[1, 2] += desired_eye_y * desired_img_height - eye_center[1]

        # warp perspective to make eyes aligned on horizontal line and scaled
        # to right size
        res = cv2.warpAffine(head, rot_mat, (desired_img_width,
                                             desired_img_width))

        # return success
        return True, res



def main():
    set_trace()
    cwd = os.getcwd()
    path = './Criminals'
    frame = cv2.imread('./Criminals/056a.jpg', cv2.CV_8UC1)
    face_casc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_casc.detectMultiScale(frame,
                                       scaleFactor=1.1,
                                       minNeighbors=3
                                       )

    for (x, y, w, h) in faces:
        # draw bounding box on frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 255, 0), 2)
    self.face_casc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    if self.face_casc.empty():
        print 'Warning: Could not load face cascade:', 'haarcascade_frontalface_default.xml'
        raise SystemExit
    set_trace()


if __name__ == "__main__":
    main()
