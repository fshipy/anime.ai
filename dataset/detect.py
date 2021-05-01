import cv2
import sys
import os
import os.path
import shutil

def detect(filename, outname, cascade_file = "./lbpcascade_animeface.xml"):
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)

    cascade = cv2.CascadeClassifier(cascade_file)
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    if image is None:
        print(filename, "is not readable!")
        return False
    h, w, c = image.shape
    #print("filename", filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    faces = cascade.detectMultiScale(gray,
                                     # detector options
                                     scaleFactor = 1.1,
                                     minNeighbors = 5,
                                     minSize = (24, 24))
    #print(faces)
    if len(faces) > 0:
        for j, face in enumerate(faces):
            x,y,w,h = face
            #print(x, y, w, h)
            high = max(0, int(y-0.15*h))
            low = max(h, int(y+0.95*h))
            left = max(0, int(x-0.1*w))
            right = max(w, int(x+1.1*w))
            output_name = str(outname) + '_' + str(j) + '.jpg'
            try:
                #cv2.imwrite(outname, image[int(y-0.1*h): int(y+0.9*h), x: x+w])
                cv2.imwrite(output_name, image[high: low, left: right])
            except:
                print("image size", image.shape)
                print("face_detected, x, y, w, h", x, y, w, h)
                print("output_scale", high, low, left, right)
                print("output_file", outname)
                print("input_file", filename)

                #print(image[int(y-0.1*h): int(y+.9*h), x: x+w])
                raise
        return True
    else:
        return False

def read_labels(path):
    label_file = open(path, 'r')
    return [line.rstrip('\n') for line in label_file.readlines()]

def main():
    labels = read_labels('labels.txt')
    for label in labels:
        print("label", label)
        output_dir = os.path.join('cropped', label)
        img_dir = os.path.join('downloads', label)
    
        ct = 0
        os.makedirs(output_dir, exist_ok=True)

        #img_dir = "downloads\\Mikoto Misaka"
        files = os.listdir(img_dir)
        for i, f in enumerate(files):
            if detect(os.path.join(img_dir, f), os.path.join(output_dir, label + '_' + str(i))):
                ct += 1
                print(ct)

main()