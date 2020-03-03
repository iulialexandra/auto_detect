from pathlib import Path
import xml.etree.ElementTree as ET
import re
from typing import Dict
import os
import numpy as np

# TODO: finish implementing

class XMLHandler:
    def __init__(self, xml_path):
        self.xml_path = Path(xml_path)
        self.root = self.__open()

        self.classes = ['__background__',  # always index 0
                        'airplane', 'antelope', 'bear', 'bicycle',
                        'bird', 'bus', 'car', 'cattle',
                        'dog', 'domestic_cat', 'elephant', 'fox',
                        'giant_panda', 'hamster', 'horse', 'lion',
                        'lizard', 'monkey', 'motorcycle', 'rabbit',
                        'red_panda', 'sheep', 'snake', 'squirrel',
                        'tiger', 'train', 'turtle', 'watercraft',
                        'whale', 'zebra']
        self.classes_map = ['__background__',  # always index 0
                            'n02691156', 'n02419796', 'n02131653', 'n02834778',
                            'n01503061', 'n02924116', 'n02958343', 'n02402425',
                            'n02084071', 'n02121808', 'n02503517', 'n02118333',
                            'n02510455', 'n02342885', 'n02374451', 'n02129165',
                            'n01674464', 'n02484322', 'n03790512', 'n02324045',
                            'n02509815', 'n02411705', 'n01726692', 'n02355227',
                            'n02129604', 'n04468005', 'n01662784', 'n04530566',
                            'n02062744', 'n02391049']

    def __open(self):
        with self.xml_path.open() as opened_xml_file:
            self.tree = ET.parse(opened_xml_file)
            return self.tree.getroot()

    def return_boxes_class_as_str(self, image_file):
        """
        Returns Dict with class name and bounding boxes.
        Key number is box number

        :return:
        """

        boxes = [image_file]
        for index, sg_box in enumerate(self.root.iter('object')):
            boxes.extend([str(sg_box.find("bndbox").find("xmin").text),
                          str(sg_box.find("bndbox").find("ymin").text),
                          str(sg_box.find("bndbox").find("xmax").text),
                          str(sg_box.find("bndbox").find("ymax").text),
                          str(self.classes[sg_box.find("name").text])])
        string = " ".join(boxes)
        description = string + "\n"
        return self.classes[sg_box.find("name").text], description


def converter(xml_folder, image_folder, output_file):
    """
    Function converts pascal voc formatted files into ODM-File format

    :param xml_files: Path to folder with xml files
    :param output_folder: Path where results files should be written
    :return:
    """
    img_files = extract_filenames(image_folder, ".jpg")
    with open(output_file, "a") as out_file:
        for f, file in enumerate(img_files):
            file = file.strip("\n")
            image_file = os.path.join(image_folder, file)
            xml_file = os.path.join(xml_folder, file.split(".")[0]+".xml")
            xml_content = XMLHandler(xml_file)
            boxes = xml_content.return_boxes_class_as_str(image_file)
            out_file.write(boxes)
    return


def sort_by_digits(unsorted_list):
    convert = lambda text: float(text) if text.isdigit() else text
    alphanum = lambda key: [convert(c) for c in re.split('([-+]?[0-9]*\.?[0-9]*)', key)]
    unsorted_list.sort(key=alphanum)
    return unsorted_list


def extract_subdirs(main_dir):
    valid_subdirs = []
    for subdir in os.listdir(main_dir):
        full_subdir = os.path.join(main_dir, subdir)
        if os.path.isdir(full_subdir):
                valid_subdirs.append(subdir)
    return valid_subdirs


def extract_filenames(main_dir, include_pattern, sort=False):
    valid_files = []
    for f in os.listdir(main_dir):
        full_path = os.path.join(main_dir, f)
        if include_pattern in full_path:
            valid_files.append(full_path)
    if sort:
        sorted_images = sort_by_digits(valid_files)
        return sorted_images
    else:
        return valid_files


def match_video_classes(video_folder_path, annot_folder_path):
    subdirs = extract_subdirs(video_folder_path)
    video_subdirs = []
    annot_subdirs = []
    for d in subdirs:
        video_subdirs.append(os.path.join(video_folder_path, d))
        annot_subdirs.append(os.path.join(annot_folder_path, d))

    print(subdirs)


if __name__ == '__main__':
    XML_FOLDER = "/mnt/data/imagenet/ILSVRC_VID/Annotations/VID"
    IMAGE_FOLDER = "/mnt/data/imagenet/ILSVRC_VID/Data/VID"
    OUTPUT_FILE_TRAIN = "/mnt/Storage/code/object detection/auto_detect/auto_collected_data/imagenet/annot_train.txt"
    OUTPUT_FILE_TEST = "/mnt/Storage/code/object detection/auto_detect/auto_collected_data/imagenet/annot_test.txt"
    match_video_classes("/mnt/data/imagenet/ILSVRC_VID/Data/VID/train/ILSVRC2015_VID_train_0000",
                        "/mnt/data/imagenet/ILSVRC_VID/Annotations/VID/train/ILSVRC2015_VID_train_0000")
