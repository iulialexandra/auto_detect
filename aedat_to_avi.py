import numpy as np
import logging
import imageio
import os
import load_aedat
from random import shuffle
import cv2
from skimage.transform import resize
from dv import LegacyAedatFile


def assemble_video(images, video_name, video_folder):
    num_ims, height, width, layers = np.shape(images)
    size = (width, height)
    video_path = os.path.join(video_folder, video_name+".avi")
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'DIVX'), 35, size)
    for im in images:
        img = np.repeat(im, 3, axis=2)
        out.write(img)
    out.release()
    return video_path


def create_frame(y_pos, x_pos, height=180, width=240):
    """Create a single frame.

    # Arguments
    y_pos : np.ndarray
        y positions
    x_pos : np.ndarray
        x positions
    """
    histrange = [(0, v) for v in (height, width)]
    # create the frame
    img, _, _ = np.histogram2d(y_pos, x_pos, bins=(height, width), range=histrange, normed=False)

    # thresholding the events
    non_zero_img = img[np.nonzero(img)]
    mean_activation = np.mean(non_zero_img)
    std_activation = np.std(non_zero_img)
    sigma = 3 * std_activation if std_activation != 0 else 1
    # clip the image
    new_img = np.clip(img / sigma, 0, 1) * 255
    return np.expand_dims(new_img.astype(int), 2)


def aedat_to_frame_list(aedat_filename, num_events, resize_scale=None):
    logging.info('Loading {}'.format(aedat_filename))

    x = []
    y = []
    timestamps = []
    polarity = []
    with LegacyAedatFile(aedat_filename) as f:
        for event in f:
            x.append(event.x)
            y.append(179 - event.y)
            timestamps.append(event.timestamp)
            polarity.append(event.polarity)

    x_addresses, y_addresses = np.array(x), np.array(y)
    num_frames = len(x_addresses) // num_events
    images = []
    for f in range(num_frames):
        frame_y = y_addresses[f * num_events: (f + 1) * num_events]
        frame_x = x_addresses[f * num_events: (f + 1) * num_events]
        images.append(create_frame(frame_y, frame_x))
    images = np.uint8(images)
    return images
    # if resize_scale is not None:
    #     resized_data = [resize(im, resize_scale, preserve_range=True, anti_aliasing=True)
    #                     for im in all_images]
    #     return np.array(resized_data)
    # else:
    #     return np.array(all_images)


if __name__ == "__main__":
    folder = "/mnt/data/roshambo/textured_background"
    contents = os.listdir(folder)
    aedats = [(file, os.path.join(folder, file)) for file in contents if ".aedat" in file]

    videos_paths = []
    for aedat_rec in aedats:
        frames = aedat_to_frame_list(aedat_rec[1], 4000)
        videos_paths.append(assemble_video(frames, aedat_rec[0].split('.')[0], folder))
