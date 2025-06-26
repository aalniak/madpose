import sys
import warnings
#warnings.simplefilter(action='ignore', category='FutureWarning')
from matplotlib import pyplot as pl
import matplotlib.pyplot as plt
import argparse
import json
import os
import time
import cv2
import gc
from scipy.spatial.transform import Rotation as R
sys.path.append('/home/snap/OGAM/match/LightGlue')
from PIL import Image
print("==============================")
print("Importing Libraries:")
import numpy as np
print("> Numpy imported.")

import torch
print("> Torch imported.")
print("> Pinhole imported.")


print("Importing LightGlue") 
from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED, DoGHardNet
from lightglue.utils import load_image, rbd
print("> LightGlue imported.")


import madpose
print("> MADPose imported.")
print("Imports done.")
print("==============================")


def get_pose_at_timestamp(timestamps, path_to_poses):
    #read the gt pose file
    gt_poses = np.loadtxt(path_to_poses)
    poses = []
    for timestamp in timestamps:
        poses.append(gt_poses[np.argmin(np.abs(gt_poses[:, 0] - timestamp))])
    return poses


def get_timestamps(path_to_imgs):
    #Names of the images are numbers, which i need to divide b 10^9 to obtain timestamps
    # This function is not used in the run loop if timestamps are loaded from a file
    # Ensure 'timestamps_f.txt' contains actual timestamp values, not image indices.
    timestamps = [int(img_name.split(".")[0]) / 10**8 for img_name in os.listdir(path_to_imgs)]
    print(timestamps)
    return timestamps


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



def run(args, path, save_name, ts, reproj_px, epipolar_px, offset, weight):
        PATH=path
        extractor = SuperPoint(max_num_keypoints=2048).eval().cuda()  # load the extractor
        matcher = LightGlue(features='superpoint').eval().cuda()  # load the matcher
        timestamps_all = np.loadtxt(ts) # Renamed to avoid conflict with loop variable

        # Initialize the global pose as an identity matrix (start at origin)
        # This will store the accumulated pose: T_world_camera
        current_global_T = np.eye(4) # 4x4 Identity matrix for [R|t]
        
        try:
            # Open the output file once at the beginning to append poses
            with open(save_name, "w") as f_out: # Use "w" to overwrite if file exists, "a" to append
                # Write the initial pose (first frame) to the file
                # Assuming the first camera is at the origin with no rotation
                initial_timestamp = timestamps_all[0] # Convert to nanoseconds
                initial_quat = R.from_matrix(current_global_T[:3, :3]).as_quat() # (x,y,z,w)
                initial_t = current_global_T[:3, 3]
                f_out.write(f"{int(initial_timestamp)} {initial_t[0]} {initial_t[1]} {initial_t[2]} {initial_quat[0]} {initial_quat[1]} {initial_quat[2]} {initial_quat[3]}\n")


                for i in range(len(timestamps_all)-1):
                    print("Iteration: ", i)
                    
                    time1 = timestamps_all[i]
                    time2 = timestamps_all[i+1]
                    
                    img_path0 = PATH+f"rgb/{int(round(timestamps_all[i]*100))}0000000.png"
                    img_path1 = PATH+f"rgb/{int(round(timestamps_all[i+1]*100))}0000000.png"
                    
                    print("Depth estimation starts")
                    with torch.no_grad():
                        if i==0:
                            depth_map0 = cv2.imread(PATH+f'depth_maps_tiff_opencv/{int(round(timestamps_all[i]*100))}0000000.tiff', cv2.IMREAD_UNCHANGED)
                            depth_map1 = cv2.imread(PATH+f'depth_maps_tiff_opencv/{int(round(timestamps_all[i+1]*100))}0000000.tiff', cv2.IMREAD_UNCHANGED)
                        else:
                            depth_map0 = depth_map1
                            depth_map1 = cv2.imread(PATH+f'depth_maps_tiff_opencv/{int(round(timestamps_all[i+1]*100))}0000000.tiff', cv2.IMREAD_UNCHANGED)
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

                    # Filter matches near borders
                    #H0, W0 = image0.shape[1], image0.shape[2]
                    #valid_matches_im0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < int(W0) - 3) & \
                    #                    (matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < int(H0) - 3)
    #
                    #H1, W1 = image1.shape[1], image1.shape[2]
                    #valid_matches_im1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < int(W1) - 3) & \
                    #                    (matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < int(H1) - 3)
    #
                    #valid_matches = valid_matches_im0 & valid_matches_im1
                    #matches_im0, matches_im1 = matches_im0[valid_matches], matches_im1[valid_matches]
                    
                    print("> Depth correspondences found.")

                    # Free VRAM
                    #del extractor
                    #del matcher
                    #gc.collect()
                    #torch.cuda.empty_cache()
                    #print("> LightGlue deleted to free VRAM.")
                    print("==============================")

                    # Reverse x and y if needed (commented out as per your original)
                    mkpts0 = matches_im0.cpu()  # [:, [1, 0]]
                    mkpts1 = matches_im1.cpu()  # [:, [1, 0]]
                    disparity_x = mkpts0[:, 0] - mkpts1[:, 0]
                    disparity_y = mkpts0[:, 1] - mkpts1[:, 1]
                    disparity_magnitude = torch.sqrt(disparity_x ** 2 + disparity_y ** 2)
                    print(f"Disparity magnitude: {disparity_magnitude.mean().item()}")
                    # Obtain depth values for matched pixels
                    depth_map0_cpu = depth_map0
                    depth_map1_cpu = depth_map1

                    # Convert coordinates to int tensors for indexing
                    mkpts0_y = mkpts0[:, 1].int()
                    mkpts0_x = mkpts0[:, 0].int()
                    mkpts1_y = mkpts1[:, 1].int()
                    mkpts1_x = mkpts1[:, 0].int()

                    depth0 = depth_map0_cpu[mkpts0_y, mkpts0_x]
                    depth1 = depth_map1_cpu[mkpts1_y, mkpts1_x]
                    
                    
                    # visualize a few matches


                    # n_viz = 25
                    # num_matches = matches_im0.shape[0]
                    # match_idx_to_viz = np.round(np.linspace(0, num_matches - 1, n_viz)).astype(int)
                    #viz_matches_im0, viz_matches_im1 = matches_im0, matches_im1
    #
                    #image_mean = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)
                    #image_std = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)
    #
                    #viz_imgs = []
                    #for k, view in enumerate([view1, view2]):
                    #    rgb_tensor = view['img'] * image_std + image_mean
                    #    viz_imgs.append(rgb_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy())
    #
                    #H0, W0, H1, W1 = *viz_imgs[0].shape[:2], *viz_imgs[1].shape[:2]
                    #img0 = np.pad(viz_imgs[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
                    #img1 = np.pad(viz_imgs[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
                    #img = np.concatenate((img0, img1), axis=1)
                    ##pl.figure()
                    ##pl.imshow(img)
                    #cmap = pl.get_cmap('jet')
                    #for i in range(len(viz_matches_im0)):
                     #   (x0, y0), (x1, y1) = viz_matches_im0[i].T, viz_matches_im1[i].T
                        # pl.plot([x0, x1 + W0], [y0, y1], '-+', color=cmap(i / (n_viz - 1)), scalex=False, scaley=False)
                      #  colors = cmap(i / (len(viz_matches_im0) - 1))
                        #plt.scatter(x0, y0, c=[colors], s=10, label=f'Match {i+1}')
                        #plt.scatter(x1 + W0, y1, c=[colors], s=10)
                        # plt.plot([x0, x1 + W0], [y0, y1], color=colors, linewidth=0.5)
                    #pl.show(block=True)



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
                    reproj_pix_thres = reproj_px
                    epipolar_pix_thres = epipolar_px
                    epipolar_weight = weight
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
                        est_config.use_shift = offset
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

                    # Now, call the error computation functions.
                    # The 'pose' object returned by HybridEstimatePoseScaleOffset is compatible
                    # as the 'model' argument for these functions.
                    #take minimum element of errors
                    # Compute reprojection errors and epipolar errors
                    # Note: These functions return lists of errors for each point.
                    # If you want to compute the minimum error, you can use .min() on the resulting lists.
                    
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
    #
                    ## Previous Camera Pose
                    #draw_camera(ax, R_prev, t_prev, color='orange', label='Before')
    #
                    ## Estimated (New) Camera Pose
                    #draw_camera(ax, R_est_relative, t_est_relative, color='blue', label='After') # Visualize relative motion
    #
                    ## Axes limits
                    #ax.set_xlim([-0.3, 0.3])
                    #ax.set_ylim([-0.3, 0.3])
                    #ax.set_zlim([-0.4, 0.1])
    #
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
    parser.add_argument("--input", type=str, default = "rgb", help="Path to input image or directory.")
    parser.add_argument("--output", type=str, default = "../../match/mast3r/out", help="Path to output directory.")
    parser.add_argument("--config-file", type=str, default="./configs/eval/vitl.json", help="Path to config file.")
    parser.add_argument("--camera-path", type=str, default='./cameras/m3ed_spot.json', help="Path to camera parameters JSON file.")
    parser.add_argument("--save", action="store_true", help="Save outputs as (colorized) PNG.")
    parser.add_argument("--save-ply", action="store_true", help="Save pointcloud as PLY.")
    parser.add_argument("--resolution-level", type=int, default=9, help="Resolution level in [0,10).", choices=list(range(10)))
    parser.add_argument("--interpolation-mode", type=str, default="bilinear", help="Output interpolation.", choices=["nearest", "nearest-exact", "bilinear"])
    args = parser.parse_args()
    print("Parsing done.")
    print("==============================")
    epipolar_px = 0.5
    reproj_px = 3
    offset = True
    offset_text = "scale_offset" if offset else "scale_only"
    camparams = True
    camparams_text = "camparams" if camparams else "nonecam"

    

    datas = {"indoor" : "/media/snap/UbuntuHDD/m3ed/spot_indoor_building_loop_data/",
             "pennoday" : "/media/snap/UbuntuHDD/m3ed/spot_outdoor_day_penno_short_loop_data/",
             "artplaza" : "/media/snap/UbuntuHDD/m3ed/spot_outdoor_day_art_plaza_loop_data/",
             "skatepark" : "/media/snap/UbuntuHDD/m3ed/spot_outdoor_day_skatepark_1_data/",
             "stairwell" : "/media/snap/UbuntuHDD/m3ed/spot_indoor_stairwell_data/",
             "stairs" : "/media/snap/UbuntuHDD/m3ed/spot_indoor_stairs_data/",
             "forest" : "/media/snap/UbuntuHDD/m3ed/spot_forest_hard_data/"}
    reproj_px_list = [3]
    epipolar_weight_list = [0.5]
    for weight in epipolar_weight_list:
        output_folder = f"./results_new/results_epipolar_weight_{weight}"
        os.makedirs(output_folder, exist_ok=True)
        for reproj_px in reproj_px_list:
            print("Starting the runs...")
            print(f"Reprojection px: {reproj_px}, Epipolar px: {epipolar_px}, Offset: {offset_text}, Camera params: {camparams_text}, Epipolar weight: {weight}" )
             # Run for each dataset in a for loop
            for dataset_name, dataset_path in datas.items():
                try:
                    start = time.time()
                    print(f"Running for {dataset_name} dataset...")
                    run(args, dataset_path, f"{output_folder}/{dataset_name}_rpe{reproj_px}_{offset_text}_{camparams_text}.txt", "timestamps_f.txt", reproj_px, epipolar_px, offset, weight)
                    end = time.time()
                    print(f"Finished processing {dataset_name} dataset in {end - start:.2f} seconds.")
                except Exception as e:
                    print(f"Error occurred while processing {dataset_name} dataset: {e}")