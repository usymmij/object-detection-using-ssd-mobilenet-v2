import os

i = 0
for root, dirs, files in os.walk("./photos"):
    for filename in files:
        os.rename('./photos/'+filename, './photos/p'+str(i)+'.jpg')
        i += 1