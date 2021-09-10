#!/usr/bin/python3

# This script converts a signle perspective depth map to a point cloud file.
# Maintainer: Julian Seuffert <julian.seuffert@etit.tu-chemnitz.de>

import getopt
import math
import os
import sys

import cv2  # python3 -m pip install --user opencv-python
import numpy as np  # python3 -m pip install --user numpy
import open3d as o3d  # python3 -m pip install --user open3d


def import_exr(file_path: str):
    import Imath
    import OpenEXR  # do not import if it is not necessary (hard to install on Windows)
    # source: https://excamera.com/articles/26/doc/intro.html
    PIXEL_TYPE = Imath.PixelType(Imath.PixelType.FLOAT)
    exr_file = OpenEXR.InputFile(file_path)
    data_window = exr_file.header()['dataWindow']
    width = data_window.max.x - data_window.min.x + 1
    height = data_window.max.y - data_window.min.y + 1
    try:
        channel = exr_file.channel('R', PIXEL_TYPE)
    except TypeError:
        channel = exr_file.channel('Y', PIXEL_TYPE)
    depth = np.frombuffer(channel, dtype=np.float32)
    depth.shape = (height, width)
    return depth


def depth_to_pc(depth: np.ndarray, focal: float, img=None):
    height, width = depth.shape
    c_x = (width - 1) / 2
    c_y = (height - 1) / 2

    # build up intrinsic (parameter) matrix K
    K = np.array([
        [focal, 0, c_x],
        [0, focal, c_y],
        [0, 0, 1]], dtype=np.float)
    K_inv = np.linalg.inv(K)

    # build up image points (not pixel values) of X_img
    x_lin_space = np.linspace(0, width - 1, width)  # 0, 1, 2, ..., w-1
    y_lin_space = np.linspace(0, height - 1, height)  # 0, 1, 2, ..., h-1

    X_img_x, X_img_y = np.meshgrid(x_lin_space, y_lin_space)

    # X_img_x:
    # 0, 1, 2, ..., w-1
    # 0, 1, 2, ..., w-1

    # X_img_y:
    #   0,   0,   0, ...,   0
    #   1,   1,   1, ...,   1
    #           ...
    # h-1, h-1, h-1, ..., h-1

    X_img_x = X_img_x.reshape((1, width * height))  # flatten the x meshgrid
    X_img_y = X_img_y.reshape((1, width * height))  # flatten the y meshgrid
    X_img_w = np.ones_like(X_img_x, dtype=np.float)  # corresponding homogeneous scaling factors for each point
    X_img = np.vstack((X_img_x, X_img_y, X_img_w))  # stacking to a point matrix

    # calculating X_norm
    X_norm = K_inv.dot(X_img)
    X_norm_inhom = X_norm[:2, :] / X_norm[2, :]

    # calculating X_cam
    z_cam = depth.reshape((1, width * height))
    focal_length_of_X_norm = 1.0
    X_cam_xy_inhom = X_norm_inhom * z_cam
    X_cam_inhom = np.vstack((X_cam_xy_inhom, z_cam))  # containing x, y and z coordinates od all 3D points

    # detecting valid depth values
    valid_1d = np.ones((1, width * height), dtype=np.bool)
    valid_1d = np.bitwise_and(z_cam > 0, np.bitwise_and(z_cam < np.inf, z_cam == z_cam))
    valid = np.zeros((3, width * height), dtype=np.bool)
    valid[0, :] = valid_1d
    valid[1, :] = valid_1d
    valid[2, :] = valid_1d

    # remove invalid values
    X_cam_inhom = X_cam_inhom[valid]
    new_width = int(X_cam_inhom.shape[0] / 3)
    X_cam_inhom = X_cam_inhom.reshape(3, new_width)
    assert len(X_cam_inhom.shape) == 2, "shape is: " + str(X_cam_inhom.shape)

    # scaling intensities (white is 255 for 8-bit images)
    # intensities must be in range [0, 1]
    factor = 1.0
    if img is not None:
        if img.dtype == np.uint8:
            factor = 1 / (2 ** 8 - 1)
        elif img.dtype == np.uint16:
            factor = 1 / (2 ** 16 - 1)
        if len(img.shape) == 3:
            img_small_b = img[:, :, 0].reshape((1, width * height))
            img_small_g = img[:, :, 1].reshape((1, width * height))
            img_small_r = img[:, :, 2].reshape((1, width * height))
            img_small = np.vstack((img_small_r, img_small_g, img_small_b))
        else:
            img_small = img[:, :].reshape((1, width * height))
            img_small = np.vstack((img_small, img_small, img_small))
    else:
        img_small = np.copy(depth)
        img_small = img_small.reshape((1, width * height))
        img_small = img_small - np.min(img_small[valid_1d])
        img_small = img_small / np.max(img_small[valid_1d]) * 0.9
        img_small = np.vstack((img_small, img_small, img_small))

    img_small = img_small[valid]
    img_small = img_small.astype(np.float64) * factor
    img_small = img_small.reshape(3, new_width)
    # print("min img_small:", np.min(img_small))
    # print("max img_small:", np.max(img_small))

    # build up Open3D poing cloud
    points = X_cam_inhom.T
    colors = img_small.T
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def help(o_stream=sys.stdout):
    print("DEPTH TO POINT CLOUD", file=o_stream)
    print("", file=o_stream)
    print("usage: python3 " + sys.argv[0] + " <options>", file=o_stream)
    print("", file=o_stream)
    print("options:", file=o_stream)
    print("{-d | --depthmap}        <depth map path>                 [mandatory]", file=o_stream)
    print("{-f | --focal} <focal length> | --fov <field of view>     [mandatory]", file=o_stream)
    print("{-h | --help}", file=o_stream)
    print("{-i | --image}           <image>", file=o_stream)
    print("{-o | --outpath}         <output path>", file=o_stream)
    print("{-t | --depththreshold}  <threshold>                      [default: 100]", file=o_stream)
    print("{-v | --visualize}", file=o_stream)
    print("", file=o_stream)
    print("parameter explanation:", file=o_stream)
    print("<depth map path>: A file path to a depth map that is stored in OpenEXR (*.exr)", file=o_stream)
    print("                  file format or a numpy structure (*.npy)", file=o_stream)
    print("<focal length>:   The camera's focal length given in pixles", file=o_stream)
    print("<field of view>:  The camera's field of view in degrees", file=o_stream)
    print("                  (currently same for x and y direction)", file=o_stream)
    print("<image>:          The optional image that is used to colorize the point cloud", file=o_stream)
    print("<output path>:    File path of the point cloud file, e.g. ./depth.ply", file=o_stream)
    print("<threshold>:      Depth values above <threshold> are not considered.", file=o_stream)


def fov_to_focal(fov: float, height: int, width: int):
    theta = fov / 2.0
    d = min(height, width) / 2
    f = d / math.tan(theta)
    return f


def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "d:f:hi:o:t:y",
                                   ["depthmap=", "focal=", "fov=", "help", "image=", "outpath=", "depththreshold=",
                                    "visualize"])
    except getopt.GetoptError as e:
        help()
        print("could not parse arguments...")
        exit(1)

    depth_map_path = None
    image_path = None
    out_path = None
    focal = None
    fov = None
    threshold = 100.0
    visualize = False

    for o, a in opts:
        if o in ("-d", "--depthmap"):
            depth_map_path = a
        elif o in ("-f", "--focal"):
            focal = float(a)
        elif o in ("--fov",):
            fov = float(a) / 180.0 * float(np.pi)
        elif o in ("-h", "--help"):
            help()
            exit(0)
        elif o in ("-i", "--image"):
            image_path = a
        elif o in ("-o", "--outpath"):
            out_path = a
        elif o in ("-t", "--depththreshold"):
            threshold = float(a)
            assert threshold > 0
        elif o in ("-v", "--visualize"):
            visualize = True
        else:
            print("unknown argument:", o)
            help(sys.stderr)
            exit(2)

    if out_path is None:
        visualize = True

    if focal is None and fov is None:
        print("Error: missing focal length [px] or field of view!", file=sys.stderr)
        help(sys.stderr)
        exit(1)

    if depth_map_path is None:
        print("Error: missing depth map path!", file=sys.stderr)
        help(sys.stderr)
        exit(1)

    ext = depth_map_path[depth_map_path.rfind('.'):]
    if ext == ".exr":
        depth = import_exr(depth_map_path)
    elif ext == ".npy":
        depth = np.load(depth_map_path)
    else:
        print("unsupported file format:", ext)
        exit(1)

    if len(depth.shape) == 3:
        depth = depth[:, :, 1]  # simply take green channel

    depth = np.copy(depth)
    depth[depth > threshold] = -1  # mask out too big values

    h, w = depth.shape

    # determine focal length if not provided
    if focal is None:
        focal = fov_to_focal(fov, h, w)
        print("Hint: focal length set to ", focal, "px")

    img = cv2.imread(image_path) if image_path is not None else None

    # print("generating point cloud")
    pcd = depth_to_pc(depth, focal, img)

    # 3D plotting
    if visualize:
        o3d.visualization.draw_geometries([pcd])
    if out_path is not None:
        o3d.io.write_point_cloud(out_path, pcd)


if __name__ == '__main__':
    main()