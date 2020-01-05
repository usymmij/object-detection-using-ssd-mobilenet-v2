import cv2

i = 20
for n in range(i):
    st = './vids/'
    st += str(n)
    st += '.mp4'
    print(st)
    vidcap = cv2.VideoCapture(st)
    success,image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite(str(n)+"frame%d.jpg" % count, image)     # save frame as JPEG file      
        success,image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1