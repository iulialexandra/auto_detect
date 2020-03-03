from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Dict
import os
import numpy as np


class XMLHandler:
    def __init__(self, xml_path):
        self.xml_path = Path(xml_path)
        self.root = self.__open()
        self.classes = {"aeroplane": 0, "bird": 1, "boat": 2, "car": 3, "cat": 4, "cow": 5,
                        "dog": 6, "horse": 7, "motorbike": 8, "train": 9}

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
        return string + "\n"


def converter(xml_folder, image_folder, set_file, output_file):
    """
    Function converts pascal voc formatted files into ODM-File format

    :param xml_files: Path to folder with xml files
    :param output_folder: Path where results files should be written
    :return:
    """
    with open(set_file) as f:
        lines = f.readlines()
    with open(output_file, "a") as out_file:
        for f, file in enumerate(lines):
            file = file.strip("\n")
            image_file = os.path.join(image_folder, file+".jpg")
            xml_file = os.path.join(xml_folder, file+".xml")
            xml_content = XMLHandler(xml_file)
            boxes = xml_content.return_boxes_class_as_str(image_file)
            out_file.write(boxes)


if __name__ == '__main__':
    XML_FOLDER = "/mnt/data/YTOdevkit/YTO/Annotations"
    IMAGE_FOLDER = "/mnt/data/YTOdevkit/YTO/JPEGImages"
    SET_FILE = "/mnt/data/YTOdevkit/YTO/ImageSets/Layout/testYTO.txt"
    OUTPUT_FILE = "/mnt/Storage/code/object detection/auto_detect/auto_collected_data/Youtube-objects/annot_test.txt"

    converter(XML_FOLDER, IMAGE_FOLDER, SET_FILE, OUTPUT_FILE)
