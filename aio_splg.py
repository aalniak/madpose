#TO DO: Border match filtering??


import sys


import warnings
#warnings.simplefilter(action='ignore', category=FutureWarning)

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as pl
import matplotlib.pyplot as plt
import argparse
import json
import os
import time
import gc
from scipy.spatial.transform import Rotation as R
import cv2  # Add OpenCV import
sys.path.append('/home/snap/OGAM/match/LightGlue')
sys.path.append('/home/snap/OGAM/depth/ml-depth-pro/src')
from depth_pro import create_model_and_transforms, load_rgb


from PIL import Image
print("==============================")
print("Importing Libraries:")
import numpy as np
print("> Numpy imported.")

import torch
print("> Torch imported.")
PATH="/media/snap/UbuntuHDD/m3ed/spot_indoor_stairwell_data/"
print("> Pinhole imported.")



print("Importing LightGlue") 
from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED, DoGHardNet
from lightglue.utils import load_image, rbd
print("> LightGlue imported.")

import madpose
print("> MADPose imported.")
print("Imports done.")
print("==============================")


from unik3d.utils.camera import (MEI, OPENCV, BatchCamera, Fisheye624, Pinhole,
                                 Spherical)
def get_pose_at_timestamp(timestamps, path_to_poses):
    #read the gt pose file
    gt_poses = np.loadtxt(path_to_poses)
    poses = []
    for timestamp in timestamps:
        poses.append(gt_poses[np.argmin(np.abs(gt_poses[:, 0] - timestamp))])
    return poses


def get_timestamps(path_to_imgs):
    #Names of the images are numbers, which i need to divide b 10^9 to obtain timestamps
    timestamps = [int(img_name.split(".")[0]) / 10**8 for img_name in os.listdir(path_to_imgs)]
    print(timestamps)
    return timestamps


def get_torch_device() -> torch.device:
    """Get the Torch device."""
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    return device

def get_rotation_and_translation(pose1, pose2):
    # Extract translation (x, y, z) and quaternion (qx, qy, qz, qw)
    x1, y1, z1, qx1, qy1, qz1, qw1 = pose1[1], pose1[2], pose1[3], pose1[4], pose1[5], pose1[6], pose1[7]
    x2, y2, z2, qx2, qy2, qz2, qw2 = pose2[1], pose2[2], pose2[3], pose2[4], pose2[5], pose2[6], pose2[7]
    
    
    # Convert quaternions to rotation matrices
    rotation1 = R.from_quat([qx1, qy1, qz1, qw1]).as_matrix()
    rotation2 = R.from_quat([qx2, qy2, qz2, qw2]).as_matrix()
    
    
    # Compute the relative rotation (rotation2 * rotation1^(-1))
    relative_rotation = np.dot(rotation2, rotation1.T)
    
    # Compute the relative translation
    translation1 = np.array([x1, y1, z1])
    translation2 = np.array([x2, y2, z2])
    
    # The relative translation is the difference in translations, rotated by the first pose's rotation
    rotated_translation1 = np.dot(rotation1.T, translation1)
    relative_translation = translation2 - translation1

    #print the relative rotation and translation
    return relative_rotation, relative_translation




def save(rgb, outputs, name, base_path, save_map=False, save_pointcloud=False):
    os.makedirs(base_path, exist_ok=True)
    depth = outputs["depth"]
    rays = outputs["rays"]
    points = outputs["points"]
    depth = depth.cpu().numpy()
    #take the average of the depth map
    
    rays = ((rays + 1) * 127.5).clip(0, 255)
    if save_map:
        np.save(os.path.join(base_path, f"{name}_depth.npy"), depth.squeeze())


def infer(model, args):
    rgb = np.array(Image.open(args.input))
    rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1)
    camera = None
    camera_path = args.camera_path
    if camera_path is not None:
        with open(camera_path, "r") as f:
            camera_dict = json.load(f)
        params = torch.tensor(camera_dict["params"])
        name = camera_dict["name"]
        assert name in ["Fisheye624", "Spherical", "OPENCV", "Pinhole", "MEI"]
        camera = eval(name)(params=params)
    
    outputs = model.infer(rgb=rgb_torch, camera=camera, normalize=True, rays=None)
    name = args.input.split("/")[-1].split(".")[0]
    #save(rgb_torch, outputs, name=name, base_path=args.output, save_map=args.save, save_pointcloud=args.save_ply)
    return outputs


def draw_camera(ax, R, t, color='b', label='Camera', scale=0.05):
    origin = t
    # Camera axes (frustum-style visualization)
    x_axis = R[:, 0] * scale
    y_axis = R[:, 1] * scale
    z_axis = R[:, 2] * scale
    
    ax.quiver(*origin, *x_axis, color='r')
    ax.quiver(*origin, *y_axis, color='g')
    ax.quiver(*origin, *z_axis, color='b')
    ax.text(*origin, label, color=color)

def unik3d_single_infer(args, model, img_path0):
    if os.path.isdir(args.input):
        # Loop through all image files in the folder
        valid_exts = (".jpg", ".jpeg", ".png", ".bmp")
        image_paths = [os.path.join(args.input, fname) for fname in sorted(os.listdir(args.input))
                       if fname.lower().endswith(valid_exts)]
            
        args_single = argparse.Namespace(**vars(args))  # Create a copy of args
        args_single.input = img_path0
        depth_map0 = infer(model, args_single)['depth']
        #bro depth_map0 is a tensor, gimme the mean of the depth map
        
        #print the mean of the depth map
        print("Depth map 0 mean: ", depth_map0.mean())
        return depth_map0

def unidepth_infer(args, model, img_path0, img_path1):
    rgb1 = np.array(Image.open(img_path0))
    rgb2 = np.array(Image.open(img_path1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("bruh")
    rgb_torch1 = torch.from_numpy(rgb1).float().permute(2, 0, 1).to(device)
    rgb_torch2 = torch.from_numpy(rgb2).float().permute(2, 0, 1).to(device)
    print("bruh2")
    #intrinsics_torch = torch.from_numpy(np.load("../../depth/UniDepth/assets/demo/intrinsics.npy"))
    #print(intrinsics_torch)
    #generate me a intrinsics matrix torch
    intrinsics_torch = torch.tensor([[1.05422725e+03, 0.00000000e+00, 6.47575998e+02],
                                   [0.00000000e+00, 1.05767516e+03, 3.65973074e+02],
                                   [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    #camera = None
    camera = Pinhole(K=intrinsics_torch.unsqueeze(0)).to(device)
    print("bruh3")
    outputs1 = model.infer(rgb=rgb_torch1, camera=camera)['depth'].squeeze(0).squeeze(0)
    print("Depth map 0 mean: ", outputs1.mean())
    outputs2 = model.infer(rgb=rgb_torch2, camera=camera)['depth'].squeeze(0).squeeze(0)
    print("Depth map 1 mean: ", outputs2.mean())

    return outputs1, outputs2


def unidepth_single_infer(args, model, img_path0):
    rgb1 = np.array(Image.open(img_path0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rgb_torch1 = torch.from_numpy(rgb1).permute(2, 0, 1).to(device)

    #intrinsics_torch = torch.from_numpy(np.load("../../depth/UniDepth/assets/demo/intrinsics.npy"))
    #print(intrinsics_torch)
    #generate me a intrinsics matrix torch
    intrinsics_torch = torch.tensor([[1.05422725e+03, 0.00000000e+00, 6.47575998e+02],
                                   [0.00000000e+00, 1.05767516e+03, 3.65973074e+02],
                                   [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    camera = Pinhole(K=intrinsics_torch.unsqueeze(0)).to(device)

    outputs1 = model.infer(rgb=rgb_torch1, camera=camera)['depth'].squeeze(0).squeeze(0)
    print("Depth map 0 mean: ", outputs1.mean())


    return outputs1

def depth_pro_infer(model,transform, img_path0, img_path1):
    """Run Depth Pro on a sample image."""
        # Load image and focal length from exif info (if found.).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    f_px = None
    rgb_image1 = Image.open(img_path0)
    rgb_image2 = Image.open(img_path1)
    # Run prediction. If `f_px` is provided, it is used to estimate the final metric depth,
    # otherwise the model estimates `f_px` to compute the depth metricness.
    #torch.tensor([5017.97], dtype=torch.float32, device=device
    prediction1 = model.infer(transform(rgb_image1), f_px=f_px) #f_px is valuable
    prediction2 = model.infer(transform(rgb_image2), f_px=f_px) #f_px is valuable
    # Extract the depth and focal length.
    depth1 = prediction1["depth"].detach().cpu().numpy().squeeze()
    depth2 = prediction2["depth"].detach().cpu().numpy().squeeze()
    print("Depth map 0 mean: ", depth1.mean())
    print("Depth map 1 mean: ", depth2.mean())

    return depth1, depth2

def depth_pro_single_infer(model, transform, img_path0):
    """Run Depth Pro on a sample image."""
        # Load image and focal length from exif info (if found.).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    f_px = None
    rgb_image1 = Image.open(img_path0)

    # Run prediction. If `f_px` is provided, it is used to estimate the final metric depth,
    # otherwise the model estimates `f_px` to compute the depth metricness.
    #torch.tensor([5017.97], dtype=torch.float32, device=device
    prediction1 = model.infer(transform(rgb_image1), f_px=f_px) #f_px is valuable

    # Extract the depth and focal length.
    depth1 = prediction1["depth"].detach().cpu().numpy().squeeze()
    print("Depth map 0 mean: ", depth1.mean())
    return depth1

def unik3d_infer(args, model, img_path0, img_path1):
    if os.path.isdir(args.input): 
        # Loop through all image files in the folder
        valid_exts = (".jpg", ".jpeg", ".png", ".bmp")
        image_paths = [os.path.join(args.input, fname) for fname in sorted(os.listdir(args.input))
                       if fname.lower().endswith(valid_exts)]

        
        args_single = argparse.Namespace(**vars(args))  # Create a copy of args
        args_single.input = img_path0
        depth_map0 = infer(model, args_single)['depth']
        args_single.input = img_path1
        depth_map1 = infer(model, args_single)['depth']
        #bro depth_map0 is a tensor, gimme the mean of the depth map
        
        #print the mean of the depth map
        print("Depth map 0 mean: ", depth_map0.mean())
        
        
        #bro depth_map1 is a tensor, gimme the mean of the depth map
        
        #print the mean of the depth map
        print("Depth map 1 mean: ", depth_map1.mean())
        print("> Depth maps inferred.")
        #del model 
        #gc.collect()
        #torch.cuda.empty_cache()
        #print("> UniK3D deleted to free VRAM.")
        print("Depth estimation done.")
        print("==============================")
        return depth_map0, depth_map1

def moge_infer(args, model, img_path0, img_path1, device):
    #find the inference time using cuda profiler
    #instead make these images batched
    input_image0 = cv2.cvtColor(cv2.imread(img_path0), cv2.COLOR_BGR2RGB)                       
    input_image0 = torch.tensor(input_image0 / 255, dtype=torch.float32, device=device).permute(2, 0, 1)    
    # Infer 
    output1 = model.infer(input_image0)
      
    input_image1 = cv2.cvtColor(cv2.imread(img_path1), cv2.COLOR_BGR2RGB)                       
    input_image1 = torch.tensor(input_image1 / 255, dtype=torch.float32, device=device).permute(2, 0, 1)    
    # Infer 
    output2 = model.infer(input_image1)
    #take time snapshot     
    return output1['depth'], output2['depth']

def moge_single_infer(args, model, img_path0, device):
    #find the inference time using cuda profiler
    #instead make these images batched
    input_image0 = cv2.cvtColor(cv2.imread(img_path0), cv2.COLOR_BGR2RGB)                       
    input_image0 = torch.tensor(input_image0 / 255, dtype=torch.float32, device=device).permute(2, 0, 1)    
    # Infer 
    output1 = model.infer(input_image0)
    
    #take time snapshot     
    return output1['depth']

def dav2_infer(args, model, img_path0, img_path1):
    rgb_image1 = Image.open(img_path0)
    rgb_image2 = Image.open(img_path1)
    depth1 = model.infer_image(np.array(rgb_image1), args.input_size)
    depth2 = model.infer_image(np.array(rgb_image2), args.input_size)
    print("Depth map 0 mean: ", depth1.mean())
    print("Depth map 1 mean: ", depth2.mean())
    return depth1, depth2

def dav2_single_infer(args, model, img_path0):
    rgb_image1 = Image.open(img_path0)
    depth1 = model.infer_image(np.array(rgb_image1), args.input_size)
    print("Depth map 0 mean: ", depth1.mean())

    return depth1

def run(args, path, save_name, ts):
        PATH=path
        timestamps = np.loadtxt(ts)
        timestamps_all = np.loadtxt(ts)
        if args.depth_model == "unik3d":
            sys.path.append('/home/snap/OGAM/depth/UniK3D')
            from unik3d.models import UniK3D
            from unik3d.utils.camera import (MEI, OPENCV, BatchCamera, Fisheye624, Pinhole,
                                 Spherical)
            print("> UniK3D imported.")
            version = args.config_file.split("/")[-1].split(".")[0]
            name = f"unik3d-{version}"
            model = UniK3D.from_pretrained(f"lpiccinelli/{name}")
            #print("> UniK3D loaded.")
            #model.resolution_level = 9
            model.interpolation_mode = "bilinear"
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device).eval()
        elif args.depth_model == "unidepth":
            sys.path.append('/home/snap/OGAM/depth/UniDepth')
            from unidepth.models import UniDepthV2
            print("> UniDepthV2 imported.")
            from unidepth.utils.camera import Pinhole
            print("> Pinhole imported.")
            model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14")
            model.interpolation_mode = "bilinear"
            model.resolution_level = 9
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device).eval()
        elif args.depth_model == "moge":
            sys.path.append('/home/snap/OGAM/depth/MoGe')
            from moge.model.v1 import MoGeModel
            print("> MoGe imported.")
            device = torch.device("cuda")
            model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(device)          
        elif args.depth_model == "DAv2":
            sys.path.append('/home/snap/OGAM/depth/Depth-Anything-V2/metric_depth')
            from depth_anything_v2.dpt import DepthAnythingV2
            print("> DepthAnythingV2 imported.")
            model_configs = {
                'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
                'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
            }
            DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
            depth_anything = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': args.max_depth})
            depth_anything.load_state_dict(torch.load(args.load_from, map_location='cpu'))
            model = depth_anything.to(DEVICE).eval()
        elif args.depth_model == "depth_pro":
            
            print("> DepthPro imported.")
            model, transform = create_model_and_transforms(
            device=get_torch_device(),
            precision=torch.half,   
            )
            model.eval()

        extractor = SuperPoint(max_num_keypoints=2048).eval().cuda()  # load the extractor
        matcher = LightGlue(features='superpoint').eval().cuda()  # load the matcher
        timestamps_all = np.loadtxt(ts)
        try:
            with open(save_name, "w") as f_out:
                initial_timestamp = timestamps_all[0] # Convert to nanoseconds
                current_global_T = np.eye(4) # 4x4 Identity matrix for [R|t]
                initial_quat = R.from_matrix(current_global_T[:3, :3]).as_quat() # (x,y,z,w)
                initial_t = current_global_T[:3, 3]
                f_out.write(f"{int(initial_timestamp)} {initial_t[0]} {initial_t[1]} {initial_t[2]} {initial_quat[0]} {initial_quat[1]} {initial_quat[2]} {initial_quat[3]}\n")
                

                for i in range(len(timestamps)-1):
                    timestamps = np.loadtxt(ts)
                    time1 = timestamps[i]
                    time2 = timestamps[i+1]

                    img_path0 = PATH+f"rgb/{int(round(timestamps[i]*100))}0000000.png"
                    img_path1 = PATH+f"rgb/{int(round(timestamps[i+1]*100))}0000000.png"

                    print("Depth estimation starts")
                    with torch.no_grad():
                        if i==0:
                            if args.depth_model == "unik3d":
                                depth_map0, depth_map1 = unik3d_infer(args, model, img_path0, img_path1)
                                depth_map0 = depth_map0.cpu().squeeze()
                                depth_map1 = depth_map1.cpu().squeeze()
                            elif args.depth_model == "unidepth":
                                depth_map0, depth_map1 = unidepth_infer(args, model, img_path0, img_path1)
                                depth_map0 = depth_map0.cpu().squeeze()
                                depth_map1 = depth_map1.cpu().squeeze()
                            elif args.depth_model == "moge":
                                depth_map0, depth_map1 = moge_infer(args, model, img_path0, img_path1, device)
                                depth_map0 = depth_map0.cpu().squeeze()
                                depth_map1 = depth_map1.cpu().squeeze()
                            elif args.depth_model == "DAv2":
                                depth_map0, depth_map1 = dav2_infer(args, model, img_path0, img_path1)
                                depth_map0 = torch.tensor(depth_map0)
                                depth_map1 = torch.tensor(depth_map1)
                            elif args.depth_model == "depth_pro":
                                depth_map0, depth_map1 = depth_pro_infer(model, transform, img_path0, img_path1)
                                depth_map0 = torch.tensor(depth_map0).cpu().squeeze()
                                depth_map1 = torch.tensor(depth_map1).cpu().squeeze()
                        else:
                            depth_map0 = depth_map1
                            if args.depth_model == "unik3d":
                                depth_map1 = unik3d_single_infer(args, model, img_path1).cpu().squeeze()
                            elif args.depth_model == "unidepth":
                                depth_map1 = unidepth_single_infer(args, model, img_path1).cpu().squeeze()
                            elif args.depth_model == "moge":
                                depth_map1 = moge_single_infer(args, model, img_path1, device).cpu().squeeze()
                            elif args.depth_model == "DAv2":
                                depth_map1 = dav2_single_infer(args, model, img_path1)
                            elif args.depth_model == "depth_pro":
                                depth_map1 = depth_pro_single_infer(model, transform, img_path1)
                    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                    '''POINT MATCHING PART: TWO IMAGES' POINT MATCHING TAKES PLACE HERE'''
                    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

                    print("Point matching starts")

                    # load each image as a torch.Tensor on GPU with shape (3,H,W), normalized in [0,1]
                    image0 = load_image(img_path0).cuda()
                    image1 = load_image(img_path1).cuda()
                    feats0 = extractor.extract(image0)
                    feats1 = extractor.extract(image1)
                    matches01 = matcher({'image0': feats0, 'image1': feats1})
                    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
                    matches = matches01['matches']
                    matches_im0 = feats0['keypoints'][matches[..., 0]]
                    matches_im1 = feats1['keypoints'][matches[..., 1]]

                    # Clear intermediate tensors
                    del image0, image1, feats0, feats1, matches01
                    torch.cuda.empty_cache()

                    print("> Point matching done.")



                    print("> Depth correspondences found.")

                    mkpts0 = matches_im0.cpu()  # [:, [1, 0]]
                    mkpts1 = matches_im1.cpu()  # [:, [1, 0]]
                    disparity_x = mkpts0[:, 0] - mkpts1[:, 0]
                    disparity_y = mkpts0[:, 1] - mkpts1[:, 1]
                    disparity_magnitude = torch.sqrt(disparity_x ** 2 + disparity_y ** 2)
                    print(f"Disparity magnitude: {disparity_magnitude.mean().item()}")
                    # Obtain depth values for matched pixels
                    if type(depth_map0) == np.ndarray:
                        depth_map0 = depth_map0
                        depth_map1 = depth_map1
                    else:
                        print(type(depth_map0))
                        depth_map0_cpu = depth_map0.cpu().squeeze()
                        depth_map1_cpu = depth_map1

                    # Convert coordinates to int tensors for indexing
                    mkpts0_y = mkpts0[:, 1].int()
                    mkpts0_x = mkpts0[:, 0].int()
                    mkpts1_y = mkpts1[:, 1].int()
                    mkpts1_x = mkpts1[:, 0].int()

                    depth0 = depth_map0_cpu[mkpts0_y, mkpts0_x]
                    depth1 = depth_map1_cpu[mkpts1_y, mkpts1_x]




                    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                    '''POSE ESTIMATION PART: TWO IMAGES' POSE ESTIMATION TAKES PLACE HERE'''
                    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                    print("Pose estimation starts")
                    options = madpose.HybridLORansacOptions()
                    options.min_num_iterations = 100
                    options.max_num_iterations = 4000
                    options.success_probability = 0.9999
                    options.random_seed = 0 # for reproducibility
                    options.final_least_squares = True
                    options.threshold_multiplier = 5.0
                    options.num_lo_steps = 6
                    # squared px thresholds for reprojection error and epipolar error
                    reproj_pix_thres = 3
                    epipolar_pix_thres = 0.5
                    epipolar_weight = 0.5
                    options.squared_inlier_thresholds = [reproj_pix_thres ** 2, epipolar_pix_thres ** 2]
                    # weight when scoring for the two types of errors
                    options.data_type_weights = [1.0, epipolar_weight]

                    est_config = madpose.EstimatorConfig()
                    # if enabled, the input min_depth values are guaranteed to be positive with the estimated depth offsets (shifts), default: True
                    est_config.min_depth_constraint = False
                    # if disabled, will model the depth with only scale (only applicable to the calibrated camera case)
                    if disparity_magnitude.mean().item() < 3:
                        print("Disparity magnitude is low, using scale only.")
                        est_config.use_shift = False
                    else:
                        print("Disparity magnitude is high, using scale and shift.")
                        est_config.use_shift = True
                    # best set to the number of PHYSICAL CPU cores
                    est_config.ceres_num_threads = 8

                    #intrinsics: [1058.1744780806393, 1058.4470113647467, 675.570437960496, 334.6606098486689]  
                    #generate K0 and K1
                    K0 = np.array([[1.11765708e+03, 0.00000000e+00, 6.57809477e+02],
                                   [0.00000000e+00, 1.22047382e+03, 3.67206231e+02],
                                   [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
                    # New camera matrix after undistortion:
                    #[[1.05422725e+03 0.00000000e+00 6.47575998e+02]
                    #[0.00000000e+00 1.05767516e+03 3.65973074e+02]
                    #[0.00000000e+00 0.00000000e+00 1.00000000e+00]]
                    K1 = np.array([[1.05422725e+03, 0.00000000e+00, 6.47575998e+02],
                                   [0.00000000e+00, 1.05767516e+03, 3.65973074e+02],
                                   [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
                    #New camera matrix after undistortion:
                    #[[821.63176275   0.         650.71112401]
                    # [  0.         825.84869162 363.35276423]
                    # [  0.           0.           1.        ]]
                    K00 = np.array([[821.63176275, 0.0, 650.71112401],
                                   [0.0, 825.84869162, 363.35276423],
                                   [0.0, 0.0, 1.0]])
                    #make mkpts0 and mkpts1 lists of np arrays
                    mkpts0 = mkpts0.tolist()
                    mkpts1 = mkpts1.tolist()

                    #make depth0 and depth1 lists of floats
                    depth0 = depth0.tolist()
                    depth1 = depth1.tolist()



                    # Format inputs correctly for HybridEstimatePoseScaleOffset
                    x0 = [np.array([[x], [y]], dtype=np.float64) for x, y in mkpts0]
                    x1 = [np.array([[x], [y]], dtype=np.float64) for x, y in mkpts1]
                    depth0 = [float(d) for d in depth0]
                    depth1 = [float(d) for d in depth1]
                    min_depth = np.array([[depth_map0_cpu.min()], [depth_map1_cpu.min()]], dtype=np.float64)

                    pose, stats = madpose.HybridEstimatePoseScaleOffset(
                                      x1, x0, # Note: Madpose expects x1, x0 for T_1_0 (transformation from 0 to 1) if using Depth1 and Depth0
                                      depth1, depth0,
                                      min_depth, 
                                      K1, K1, options, est_config
                                  )

                    R_est_relative, t_est_relative = pose.R(), pose.t() # These are the relative R and t from cam0 to cam1
                    # scale and offsets of the affine corrected depth maps
                    s_est, o0_est, o1_est = pose.scale, pose.offset0, pose.offset1



                    estimator_scale_only = madpose.madpose.HybridPoseEstimator(
                        x0=x1,                          # Note: Order is x0, x1 not x1, x0 like the RANSAC call
                        x1=x0,
                        depth0=depth1,
                        depth1=depth0,
                        min_depth=min_depth,
                        K0=K1,
                        K1=K1,
                        sampson_squared_weight=0.5,
                        squared_inlier_thresholds=[reproj_pix_thres ** 2, epipolar_pix_thres ** 2],
                        est_config=est_config
                    )
                    print("HybridPoseEstimator instantiated successfully.")


                    reproj_errors0 = estimator_scale_only.compute_reprojection_errors_0(pose)

                    #least_reproj_errors0 = min(reproj_errors0)
                    reproj_errors1 = estimator_scale_only.compute_reprojection_errors_1(pose)

                    #least_reproj_errors1 = min(reproj_errors1)
                    epipolar_errors = estimator_scale_only.compute_epipolar_errors(pose)

                    #least_epipolar_errors = min(epipolar_errors)
                    all_errors_scale_only = estimator_scale_only.compute_all_errors(pose)


                    print(f"Best Model Score: {stats.best_model_score}")
                    print(f"Best number of Inliers: {stats.best_num_inliers}")
                    print(f"Best Solver Type: {stats.best_solver_type}")
                    print(f"Number of iterations per solver: {stats.num_iterations_per_solver}")
                    print(f"Number of iterations total: {stats.num_iterations_total}")
                    print(f"Number LO iterations: {stats.number_lo_iterations}")
                    print("\n--- Computed Errors (HybridPoseEstimatorScaleOnly) ---")
                    print(len(reproj_errors0), "reprojection errors computed for 0->1") 
                    #print(f"Reprojection Errors 0->1, least:: {least_reproj_errors0}")
                    print(len(reproj_errors1), "reprojection errors computed for 1->0")
                    #print(f"Reprojection Errors 1->0, least): {least_reproj_errors1}")
                    print(len(epipolar_errors), "epipolar errors computed")
                    #print(f"Epipolar Errors, least): {least_epipolar_errors}")
                    print(f"All Combined Errors (first 5 tuples): {all_errors_scale_only[:5]}")
                    print(f"Total points evaluated: {len(all_errors_scale_only)}")
                    # rotation and translation of the estimated pose

                    print(">>Number of matches: ", len(matches_im0))
                    print(">>RANSAC Inliers: ", stats.best_num_inliers)
                    print(">>RANSAC Inlier Ratio: ", stats.inlier_ratios)


                    print("> Estimated pose (relative): ", pose)
                    print("> Estimated scale: ", s_est)
                    print("> Estimated offset0: ", o0_est)
                    print("> Estimated offset1: ", o1_est)
                    print("> Estimated rotation (relative): ", R_est_relative)
                    print("> Estimated translation (relative): ", t_est_relative)
                    print("==============================")
                    #write the estimated rotation and translation into results.txt

                    #get timestamps
                    def get_line_of_timestamp(timestamp):
                        #read the timestamps.txt file
                        #gt_modified.txt has the ts, x, y, z, qx, qy, qz, qw
                        timestamps_gt = np.loadtxt(PATH+"gt.txt")[:,0] # Renamed to avoid conflict
                        #get the line of the timestamp that is closer to the input timestamp
                        line = np.argmin(np.abs(timestamps_gt - timestamp))
                        timestamps_gt  = np.loadtxt(PATH+"gt.txt")
                        #print  the entire line 
                        return timestamps_gt[line]
                    def pose_line_to_Twc(line):
                        #i have a list for these items, so i need to unpack them
                        ts, x, y, z, qx, qy, qz, qw = line
                        R_wc = R.from_quat([qx, qy, qz, qw]).as_matrix()
                        t_wc = np.array([x, y, z])
                        Twc = np.eye(4)
                        Twc[:3, :3] = R_wc
                        Twc[:3, 3] = t_wc
                        return ts, Twc
                    def relative_T12(Twc1, Twc2):
                        Rwc1, twc1 = Twc1[:3,:3], Twc1[:3,3]
                        Rwc2, twc2 = Twc2[:3,:3], Twc2[:3,3]
                        R12 = Rwc1.T @ Rwc2
                        t12 = Rwc1.T @ (twc2 - twc1)
                        return R12, t12

                    print(i)
                    print(timestamps_all[i])
                    ts1_gt, Twc1_gt = pose_line_to_Twc(get_line_of_timestamp(timestamps_all[i]))
                    ts2_gt, Twc2_gt = pose_line_to_Twc(get_line_of_timestamp(timestamps_all[i+1]))
                    R12_gt, t12_gt = relative_T12(Twc1_gt, Twc2_gt)
                    print("> GT rotation: ", R12_gt)
                    print("> GT translation: ", t12_gt)

                    # Apply the relative pose to the current global pose
                    # T_world_camera_new = T_world_camera_old @ T_camera_old_camera_new (relative_pose)
                    T_relative_4x4 = np.eye(4)
                    T_relative_4x4[:3, :3] = R_est_relative
                    T_relative_4x4[:3, 3] = t_est_relative
                    if np.linalg.norm(t_est_relative) < 1:
                        current_global_T = current_global_T @ T_relative_4x4
                        # Extract R and t from the updated global pose
                        R_est_global = current_global_T[:3, :3]
                        t_est_global = current_global_T[:3, 3]
                        # Convert rotation matrix R_est_global to quaternion
                        r_est_quat_global = R.from_matrix(R_est_global).as_quat() # (x, y, z, w)
                        # TUM format expects timestamp in nanoseconds
                        current_timestamp = timestamps_all[i+1]

                        # TUM format: timestamp tx ty tz qx qy qz qw
                        line_to_write = f"{float(current_timestamp)} {t_est_global[0]} {t_est_global[1]} {t_est_global[2]} {r_est_quat_global[0]} {r_est_quat_global[1]} {r_est_quat_global[2]} {r_est_quat_global[3]}\n"
                        f_out.write(line_to_write)

                    # --- Assume Previous Pose is Identity (Origin) ---
                    # R_prev = np.eye(3) # These are for visualization, keep them as relative if needed for plotting
                    # t_prev = np.zeros(3)
                    # --- Visualize ---
                    #fig = plt.figure()
                    #ax = fig.add_subplot(111, projection='3d')

                    ## Previous Camera Pose
                    #draw_camera(ax, R_prev, t_prev, color='orange', label='Before')

                    ## Estimated (New) Camera Pose
                    #draw_camera(ax, R_est_relative, t_est_relative, color='blue', label='After') # Visualize relative motion

                    ## Axes limits
                    #ax.set_xlim([-0.3, 0.3])
                    #ax.set_ylim([-0.3, 0.3])
                    #ax.set_zlim([-0.4, 0.1])

                    #ax.set_xlabel('X')
                    #ax.set_ylabel('Y')
                    #ax.set_zlabel('Z')
                    #ax.set_title('Camera Pose Transformation')
                    #plt.show()

                    # Clear depth maps after use
                    del depth_map0
                    torch.cuda.empty_cache()

            # f_out will be automatically closed when exiting the 'with' block
        finally:
            # Clean up at the end
            del extractor, matcher
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    
        #get the two images
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    '''DEPTH ESTIMATION PART: TWO IMAGES' DEPTH ESTIMATION TAKES PLACE HERE'''
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    parser = argparse.ArgumentParser(description='Inference script', conflict_handler='resolve')
    parser.add_argument("--input", type=str, default = PATH+"rgb", help="Path to input image or directory.")
    parser.add_argument("--output", type=str, default = "../../match/mast3r/out", help="Path to output directory.")
    parser.add_argument("--config-file", type=str, default="./configs/eval/vitl.json", help="Path to config file.")
    parser.add_argument("--camera-path", type=str, default='./cameras/m3ed_spot.json', help="Path to camera parameters JSON file.")
    parser.add_argument("--save", action="store_true", help="Save outputs as (colorized) PNG.")
    parser.add_argument("--save-ply", action="store_true", help="Save pointcloud as PLY.")
    parser.add_argument("--resolution-level", type=int, default=9, help="Resolution level in [0,10).", choices=list(range(10)))
    parser.add_argument("--interpolation-mode", type=str, default="bilinear", help="Output interpolation.", choices=["nearest", "nearest-exact", "bilinear"])
    parser.add_argument("--depth-model", type=str, default="unidepth", help="Depth model to use.", choices=["unik3d", "unidepth", "moge", "DAv2", "depth_pro"])
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--encoder', type=str, default='vitb', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--load-from', type=str, default='../../depth/Depth-Anything-V2/metric_depth/checkpoints/depth_anything_v2_metric_hypersim_vitb.pth')
    parser.add_argument('--max-depth', type=float, default=20)
    args = parser.parse_args()
    print("Parsing done.")
    print("==============================")
    args.depth_model = "unidepth"
    try:
        run(args, "/media/snap/UbuntuHDD/m3ed/spot_outdoor_day_skatepark_1_data/", "./results_mde_models/skatepark_rpe3_scale_offset_UD_evo_sot.txt", "timestamps_f.txt")    
    except Exception as e:
        print(e)
    #args.depth_model = "unidepth"
    #try:
    #    run(args, "/media/snap/UbuntuHDD/m3ed/spot_indoor_building_loop_data/", "./results_mde_models/indoor_rpe3_scale_only_UD_evo.txt", "timestamps_f.txt")    
    #except Exception as e:
    #    print(e)
    #args.depth_model = "moge"
    #try:
    #    run(args, "/media/snap/UbuntuHDD/m3ed/spot_indoor_building_loop_data/", "./results_mde_models/indoor_rpe3_scale_only_moge_evo.txt", "timestamps_f.txt")    
    #except Exception as e:
    #    print(e)
    #try:
    #    run(args, "/media/snap/UbuntuHDD/m3ed/spot_outdoor_day_penno_short_loop_data/", "./results/pennoday_rpe2_scale_offset_camparams.txt", "timestamps_f.txt")    
    #except Exception as e:
    #    print(e)
    #try:
    #    run(args, "/media/snap/UbuntuHDD/m3ed/spot_outdoor_day_art_plaza_loop_data/", "./results/artplaza_rpe2_scale_offset_camparams.txt", "timestamps_f.txt")    
    #except Exception as e:
    #    print(e)
    #
    #
    #try:
    #    run(args, "/media/snap/UbuntuHDD/m3ed/spot_outdoor_day_skatepark_1_data/", "./results/skatepark_rpe2_scale_offset_camparams.txt", "timestamps_f.txt")    
    #except Exception as e:
    #    print(e)
    
    #try:
    #    run(args, "/media/snap/UbuntuHDD/m3ed/spot_indoor_stairwell_data/", "./results/stairwell_rpe2_scale_offset_camparams.txt", "timestamps_f.txt")    
    #except Exception as e:
    #    print(e)
    #
    #run(args, "/media/snap/UbuntuHDD/m3ed/spot_outdoor_day_skatepark_1_data/", "./results_mde_models/skatepark_rpe3_scale_offset_DP_evo.txt", "timestamps_f.txt")    
    
    #try:
    #    run(args, "/media/snap/UbuntuHDD/m3ed/spot_forest_hard_data/", "./results/forest_rpe2_scale_offset_camparams.txt", "timestamps_f.txt")    
    #except Exception as e:
    #    print(e)
    
    

    
    
