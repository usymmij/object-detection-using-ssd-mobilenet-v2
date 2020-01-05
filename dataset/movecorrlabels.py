import re
import os

for root, dirs, files in os.walk("./_test/images"):
    for filename in files:
        name = re.search('(.*)(\.jpg)', filename)
        name = name.group(1)
        os.rename('./labels/'+name+'.xml', './_test/labels/'+name+'.xml')