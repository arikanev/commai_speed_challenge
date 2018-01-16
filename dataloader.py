import cv2
import sys
import pickle as pkl
import numpy as np
from timeit import default_timer as timer

trainvid = cv2.VideoCapture('train.mp4')
testvid = cv2.VideoCapture('test.mp4')

# y train loading

# train label list
Y = []

with open('train.txt') as train_labels:
    for line in train_labels:
        Y.append(line)

# x train loading

# train inp list
X = []

success_flag = True
while success_flag:

    success_flag, image = trainvid.read()
    X.append(image)

    percent = float(len(X) / 20400) * 100

    arrow = '-' * int(round(len(X) / 1000)-1) + '>'
    spaces = ' ' * (20 - len(arrow))

    sys.stdout.write("\rPercent: [{0}] {1}%".format(arrow +
                     spaces, int(round(percent))))
    sys.stdout.flush()

    sys.stdout.write(" " + str(len(X)) + " out of " + " 20400 files loaded")
 
    sys.stdout.flush()

# x:y dict creation

# x:y dict
train = dict()

train['data'] = X
train['labels'] = np.array(Y)

print(train['data'])
print(train['labels'])

# save as pkl
output = open('train.pkl', 'wb')
pkl.dump(train, output)
output.close()

def test():
    train = pkl.load(open("train.pkl", "rb"))
    for i in range(len(train['data'])):
        print(train['data'][i])

if __name__ == '__main__':
    test()
