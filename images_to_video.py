import re
import cv2
import numpy as np
import logging
import imageio
import os
from random import shuffle
from skimage.transform import resize


def sort_by_digits(unsorted_list):
    convert = lambda text: float(text) if text.isdigit() else text
    alphanum = lambda key: [convert(c) for c in re.split('([-+]?[0-9]*\.?[0-9]*)', key)]
    unsorted_list.sort(key=alphanum)
    return unsorted_list


def extract_subdirs(main_dir, include_pattern, exclude_folder=None, exclude_pattern=None):
    valid_subdirs = []
    for subdir in os.listdir(main_dir):
        full_subdir = os.path.join(main_dir, subdir)
        if exclude_pattern:
            if (os.path.isdir(full_subdir)
                    and include_pattern in str(full_subdir)
                    and exclude_pattern not in str(full_subdir)
                    and str(full_subdir) != exclude_folder):
                valid_subdirs.append(full_subdir)
        else:
            if (os.path.isdir(full_subdir)
                    and include_pattern in str(full_subdir)
                    and str(full_subdir) != exclude_folder):
                valid_subdirs.append(full_subdir)
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


def avi_to_frame_list(avi_filename, video_limit=-1, resize_scale=None):
    """Creates a list of frames starting from an AVI movie.

    Parameters
    ----------

    avi_filename: name of the AVI movie
    gray: if True, the resulting images are treated as grey images with only
          one channel. If False, the images have three channels.
    """
    logging.info('Loading {}'.format(avi_filename))
    try:
        vid = imageio.get_reader(avi_filename, 'ffmpeg')
    except IOError:
        logging.error("Could not load meta information for file %".format(avi_filename))
        return None
    data = [im for im in vid.iter_data() if np.sum(im) > 2000]
    if data is None:
        return
    else:
        shuffle(data)
        video_limit = min(len(data), video_limit)
        assert video_limit != 0, "The video limit is 0"
        data = data[:video_limit]
        expanded_data = [np.expand_dims(im[:, :, 0], 2) for im in data]
        if resize_scale is not None:
            expanded_data = [resize(im, resize_scale, preserve_range=True) for im in expanded_data]
        logging.info('Loaded frames from {}.'.format(avi_filename))
        return expanded_data


def assemble_video(image_path, video_out_folder):
    class_filenames = extract_filenames(image_path, ".jpg", sort=True)
    class_name = image_path.split("/")[-2]
    img = cv2.imread(class_filenames[0])
    height, width, layers = img.shape
    size = (width, height)
    video_path = os.path.join(video_out_folder, class_name + ".avi")
    out = cv2.VideoWriter(video_path,
                          cv2.VideoWriter_fourcc(*'DIVX'), 35, size)
    for f in class_filenames:
        img = cv2.imread(f)
        out.write(img)
    out.release()
    return video_path


def convert_TLP_to_video():
    object_image_folders = ["/mnt/data/TLP/Alladin/img",
                            "/mnt/data/TLP/Bike/img",
                            "/mnt/data/TLP/Boat/img",
                            "/mnt/data/TLP/Drone1/img",
                            "/mnt/data/TLP/Elephants/img",
                            "/mnt/data/TLP/CarChase1/img",
                            "/mnt/data/TLP/Lion/img",
                            "/mnt/data/TLP/Parakeet/img",
                            "/mnt/data/TLP/PolarBear1/img",
                            "/mnt/data/TLP/ZebraFish/img"]

    videos_paths = []
    for obj_class in object_image_folders:
        videos_paths.append(assemble_video(obj_class,
                            "/mnt/Storage/code/object detection/auto_collected_data/TLP/videos"))
    return videos_paths


if __name__ == "__main__":
    print(convert_TLP_to_video())
