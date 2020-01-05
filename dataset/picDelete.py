import os

i = 0
for root, dirs, files in os.walk("./photos"):
    for filename in files:
        if i == 20:
            i = 0
            continue
        else:
            i += 1
            os.remove('./photos/'+filename)