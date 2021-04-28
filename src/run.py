import sys
import cv2
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
output_path = './graded_images/'

from SingleDetectionAndRecognition import classify

arguments = len(sys.argv)
if arguments == 2:
    _, path = sys.argv
    image = cv2.imread(os.path.join(path))
    out_image = classify(image)
    fn = path.split('/')[-1]
    print('writing to', output_path + "output_{}".format(fn))
    cv2.imwrite(output_path + "output_{}".format(fn), out_image)
else:
    print("Error - invalid number of arguments - Usage: python run.py path")
