
# coding: utf-8
# In[1]:

#!/usr/bin/env python

"""

Description::


"""

import cv2
from detectors_adapt import FaceDetector
import os
import sys
import pca
import pandas as pd
import time
import classifier_nn
import components


def set_trace():
    """A Poor mans break point"""
    # without this in iPython debugger can generate strange characters.
    from IPython.core.debugger import Pdb
    Pdb().set_trace(sys._getframe().f_back)


def combine_class_arrays(list1, list2):
    df1 = pd.DataFrame(list1)
    df2 = pd.DataFrame(list2)
    df = pd.concat([df1, df2])
    return df


def main():
    st = time.time()
    sample1 = []
    labels1 = []
    sample2 = []
    labels2 = []
    for filename in os.listdir('C:\Users\\annye\Documents\FacialExpressions\\crim'):
        image = cv2.imread('C:\Users\\annye\Documents\FacialExpressions\\crim\\' + filename)
        grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # cv2.imwrite('bad2_grey.png', grey_image)
        face_casc = 'haarcascade_frontalface_default.xml'
        left_eye_casc = 'haarcascade_lefteye_2splits.xml'
        right_eye_casc = 'haarcascade_righteye_2splits.xml'

        faces = FaceDetector(face_casc, left_eye_casc, right_eye_casc)
        success1, frame, head = faces.detect(image)
        if success1:
            success2, head = faces.align_head(head)
            if success2:
                sample1.append(head.flatten())
                # labels1.append('criminal')
                labels1.append(1)
            else:
                pass
        else:
            pass

    for filename in os.listdir('C:\Users\\annye\Documents\FacialExpressions\\non'):
        image = cv2.imread('C:\Users\\annye\Documents\FacialExpressions\\non\\' + filename)
        grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # cv2.imwrite('bad2_grey.png', grey_image)
        face_casc = 'haarcascade_frontalface_default.xml'
        left_eye_casc = 'haarcascade_lefteye_2splits.xml'
        right_eye_casc = 'haarcascade_righteye_2splits.xml'

        faces = FaceDetector(face_casc, left_eye_casc, right_eye_casc)
        success1, frame, head = faces.detect(image)
        if success1:
            success2, head = faces.align_head(head)
            if success2:
                sample2.append(head.flatten())
                # labels2.append('Non-criminal')
                labels2.append(0)
            else:
                pass
        else:
            pass
    df1 = combine_class_arrays(sample1, sample2)
    set_trace()
    print 'length (Criminals): ', len(sample1)
    print 'length (Non-Criminals): ', len(sample2)
    
    y = combine_class_arrays(labels1, labels2)
    y = y.rename(columns={0: 'Class'})
    components.number_pcs = df1
    x_reduced = pca.perform_pca(df1, y, st)
    x_reduced = pd.DataFrame(x_reduced)
    y = y.reset_index(drop=True)
    full = pd.concat([x_reduced, y], axis=1)
    x, y, validation = classifier_nn.get_traintest_validation_split(full)
    classifier_nn.perform_logistic_regression(x, y)
    # classifier_nn.manual_ann(x, y, validation)
    # classifier_nn.ann_k_fold_validation(x, y)
    # classifier_nn.ann_with_dropout(x, y)
    print ('Process execution took {0} seconds'.format(time.time() - st))
    

if __name__ == "__main__":
    main()



