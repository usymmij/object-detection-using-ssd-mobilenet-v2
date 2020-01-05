import os
import xml.etree.ElementTree as ET
import tensorflow as tf
from tensorflow.models.research.object_detection.utils import dataset_util

# This are the path to the datasets and to the output files.
# NEED TO BE UPDATED IN CASE THE DATASET CHANGES
PATH_TEST = "./_test/"
PATH_RECORD_TEST = "test1.record"
PATH_TRAIN = "./_train/"
PATH_RECORD_TRAIN = "train1.record"

# This function defines the different classes the dataset has and return a different number per each.
# NEED TO BE UPDATED IN CASE THE DATASET CHANGES
def class_text_to_int(row_label):
    if row_label == 'moon':
        return 1
    else:
        none

# Reads the xml and the images, and create the tf records files. 
def xml_to_tf(path_input, path_output):
    xml_list = []
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']

    writer = tf.io.TFRecordWriter(path_output)

    files = os.listdir(path_input)
    for file in files:
        if file.endswith(".xml"):
            xmlFile = path_input + file

            tree = ET.parse(xmlFile)
            root = tree.getroot()
            
            filename = root[1].text
            width = int(root[4][0].text)
            height = int(root[4][1].text)

            xmins = []
            xmaxs = []
            ymins = []
            ymaxs = []
            classes_text = []
            classes = []
            

            for member in root.findall('object'):
                beer = member[0].text
                xmin = int(member[4][0].text)
                ymin = int(member[4][1].text)
                xmax = int(member[4][2].text)
                ymax = int(member[4][3].text)
                
                xmins.append(xmin/width)
                xmaxs.append(xmax/width)
                ymins.append(ymin/height)
                ymaxs.append(ymax/height)
                classes_text.append(beer.encode('utf8'))
                classes.append(class_text_to_int(beer))

            with tf.io.gfile.GFile(os.path.join(path_input, '{}'.format(filename)), 'rb') as fid:
                encoded_jpg = fid.read()
                img = encoded_jpg
            tf_example = tf.train.Example(features=tf.train.Features(feature={
                'img': dataset_util.bytes_feature(img),
                'height': dataset_util.int64_feature(height),
                'width': dataset_util.int64_feature(width),
                'filename': dataset_util.bytes_feature(filename.encode('utf8')),
                'source_id': dataset_util.bytes_feature(filename.encode('utf8')),
                'xmin': dataset_util.float_list_feature(xmins),
                'xmax': dataset_util.float_list_feature(xmaxs),
                'ymin': dataset_util.float_list_feature(ymins),
                'ymax': dataset_util.float_list_feature(ymaxs),
                'text': dataset_util.bytes_list_feature(classes_text),
                'label': dataset_util.int64_list_feature(classes),
            }))
            
            writer.write(tf_example.SerializeToString())
    writer.close()             
    output_path = os.path.join(os.getcwd(), path_output)
    print('Successfully created the TFRecords: {}'.format(output_path))

xml_to_tf(PATH_TEST, PATH_RECORD_TEST)
xml_to_tf(PATH_TRAIN, PATH_RECORD_TRAIN)