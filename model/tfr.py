import tensorflow as tf
import xml.etree.ElementTree as ET
from object_detection.utils import dataset_util
import os
import re

IMAGE_FORMAT = b'jpg'

flags = tf.app.flags

flags.DEFINE_string('train_path', '../dataset/_train/', 'fuck you and ur help, i don`t care')
flags.DEFINE_string('test_path', '../dataset/_test/', 'did you know that dogs have penis bones')
flags.DEFINE_string('test_record', './_test.record', 'empty_string')
flags.DEFINE_string('train_record', './_train.record', 'wooloomooloo')
FLAGS = flags.FLAGS

def class_text_to_int(row_label):
    if row_label == 'moon':
        return 1

def create_tfrecord(path_in, path_out):
    writer = tf.io.TFRecordWriter(path_out)
    files = os.listdir(path_in)
    for file in files:
        if file.endswith(".xml"):

            xml_path = path_in + file
            tree = ET.parse(xml_path)
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

            with tf.io.gfile.GFile(os.path.join(path_in, '{}'.format(filename)), 'rb') as fid:
                encoded_jpg = fid.read()

            data_unit = tf.train.Example(features=tf.train.Features(feature={
                'image/height': dataset_util.int64_feature(height),
                'image/width': dataset_util.int64_feature(width),
                'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
                'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
                'image/encoded': dataset_util.bytes_feature(encoded_jpg),
                'image/format': dataset_util.bytes_feature(IMAGE_FORMAT),
                'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
                'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
                'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
                'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
                'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
                'image/object/class/label': dataset_util.int64_list_feature(classes),
            }))
            writer.write(data_unit.SerializeToString()) 
    writer.close()             
    output_path = os.path.join(os.getcwd(), path_out)
    print('Successfully created the TFRecords: {}'.format(output_path))


create_tfrecord(FLAGS.test_path, FLAGS.test_record)
create_tfrecord(FLAGS.train_path, FLAGS.train_record)