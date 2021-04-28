import sys
import cv2
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
output_path = './graded_images/'

from SingleDetectionAndRecognition import detect_and_classify

arguments = len(sys.argv)
if arguments == 2:
    path = sys.argv[1]
    image = cv2.imread(os.path.join(path))
    out_image = detect_and_classify(image)
    print('writing to', output_path + "output_{}".format(path))
    cv2.imwrite(output_path + "output_{}".format(path), out_image)
else:
    print("Error - invalid number of arguments - Usage: python run.py path")
