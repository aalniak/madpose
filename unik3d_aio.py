import sys
## Add all paths before any imports
#sys.path.append('/home/arda/OGAM/UniK3D')
sys.path.append('/home/arda/OGAM/mast3r')
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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

sys.path.append('/home/arda/OGAM/UniK3D')
from PIL import Image
print("==============================")
print("Importing Libraries:")
import numpy as np
print("> Numpy imported.")

import torch
print("> Torch imported.")


from unik3d.models import UniK3D
print("> UniK3D imported.")

print("Importing MASt3R")
from mast3r.model import AsymmetricMASt3R
print("> MASt3R imported.")

from dust3r.inference import inference
from dust3r.utils.image import load_images
print("> Dust3R inference and image loader imported.")

from mast3r.fast_nn import fast_reciprocal_NNs
print("> Fast reciprocal NNs imported.")

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


def run(args):
        timestamps = np.loadtxt("timestamps_f.txt")
        unik3d = False
        mast3r = False
        
        for i in range(len(timestamps)-1):
            timestamps = np.loadtxt("timestamps_f.txt")
            time1 = timestamps[i]
            time2 = timestamps[i+1]
            
            img_path0 = f"../mast3r/asd/{int(round(timestamps[i]*10))}0000000.png"
            img_path1 = f"../mast3r/asd/{int(round(timestamps[i+1]*10))}0000000.png"
            
            #these are float numbers, such as 0.4, which i want to multiply with 10^8 for naming the images
            
    
            print("Depth estimation starts")
            if unik3d and 'model' in locals():
                print("Already loaded UniK3D.")
            else:
                version = args.config_file.split("/")[-1].split(".")[0]
                name = f"unik3d-{version}"
                model = UniK3D.from_pretrained(f"lpiccinelli/{name}")
                print("> UniK3D loaded.")
                model.resolution_level = args.resolution_level
                model.interpolation_mode = args.interpolation_mode
                unik3d = True

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = model.to(device).eval()


            #timestamps = np.loadtxt("timestamps_f.txt")

            if os.path.isdir(args.input):
                # Loop through all image files in the folder
                valid_exts = (".jpg", ".jpeg", ".png", ".bmp")
                image_paths = [os.path.join(args.input, fname) for fname in sorted(os.listdir(args.input))
                               if fname.lower().endswith(valid_exts)]


                
                args_single = argparse.Namespace(**vars(args))  # Create a copy of args
                args_single.input = img_path0
                depth_map0 = infer(model, args_single)['depth']
                #bro depth_map0 is a tensor, gimme the mean of the depth map
                depth_map0 = depth_map0.cpu().squeeze()
                #print the mean of the depth map
                print("Depth map 0 mean: ", depth_map0.mean())
                args_single.input = img_path1
                depth_map1 = infer(model, args_single)['depth']
                #bro depth_map1 is a tensor, gimme the mean of the depth map
                depth_map1 = depth_map1.cpu().squeeze()
                #print the mean of the depth map
                print("Depth map 1 mean: ", depth_map1.mean())

                
            
            print("> Depth maps inferred.")
            del model 
            gc.collect()
            torch.cuda.empty_cache()
            print("> UniK3D deleted to free VRAM.")
            print("Depth estimation done.")
            print("==============================")


            ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
            '''POINT MATCHING PART: TWO IMAGES' POINT MATCHING TAKES PLACE HERE'''
            ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
            print("Point matching starts")
            device = 'cuda'
            schedule = 'cosine'
            lr = 0.01
            niter = 1000
            model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
            print("> MASt3R loaded.")
            # you can put the path to a local checkpoint in model_name if needed
            if mast3r and 'mast3r_model' in locals():
                print("Already loaded MASt3R.")
            else:
                mast3r_model = AsymmetricMASt3R.from_pretrained(model_name).to(device)
                mast3r = True
            images = load_images([img_path0, img_path1], size=1280)
            output = inference([tuple(images)], mast3r_model, device, batch_size=1, verbose=False)

            # at this stage, you have the raw dust3r predictions
            view1, pred1 = output['view1'], output['pred1']
            view2, pred2 = output['view2'], output['pred2']

            desc1, desc2 = pred1['desc'].squeeze(0).detach(), pred2['desc'].squeeze(0).detach()

            # find 2D-2D matches between the two images
            matches_im0, matches_im1 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=8,
                                                           device=device, dist='dot', block_size=2**13)
            num_matches = matches_im0.shape[0]
            print("Number of matches from Mast3R point matching: ", num_matches)
            max_matches = 7000

            # Take first max_matches matches randomly
            if num_matches > max_matches:
                indices = np.random.choice(num_matches, max_matches, replace=False)
                matches_im0 = matches_im0[indices]
                matches_im1 = matches_im1[indices]
                num_matches = max_matches
            print("Number of matches after taking first max_matches matches randomly: ", num_matches)

            print("> Point matching done.")
            # ignore small border around the edge
            H0, W0 = view1['true_shape'][0]
            valid_matches_im0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < int(W0) - 3) & (
                matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < int(H0) - 3)

            H1, W1 = view2['true_shape'][0]
            valid_matches_im1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < int(W1) - 3) & (
                matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < int(H1) - 3)

            valid_matches = valid_matches_im0 & valid_matches_im1
            matches_im0, matches_im1 = matches_im0[valid_matches], matches_im1[valid_matches]

            print("> Depth correspondances found.")
            del mast3r_model
            gc.collect()
            torch.cuda.empty_cache()
            print("> MASt3R deleted to free VRAM.")
            print("==============================")
            #create two variables for match pixels

            #reverse x and y
            mkpts0 = matches_im0#[:, [1, 0]]
            mkpts1 = matches_im1#[:, [1, 0]]


            #obtain depth values for match pixels
            depth_map0_cpu = depth_map0.cpu().squeeze() 
            #print matches_im0's max value in y
            depth_map1_cpu = depth_map1.cpu().squeeze() 

            depth0 = depth_map0_cpu[mkpts0[:, 1], mkpts0[:, 0]]
            depth1 = depth_map1_cpu[mkpts1[:, 1], mkpts1[:, 0]]

            # visualize a few matches


            # n_viz = 25
            # num_matches = matches_im0.shape[0]
            # match_idx_to_viz = np.round(np.linspace(0, num_matches - 1, n_viz)).astype(int)
            viz_matches_im0, viz_matches_im1 = matches_im0, matches_im1

            image_mean = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)
            image_std = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)

            viz_imgs = []
            for k, view in enumerate([view1, view2]):
                rgb_tensor = view['img'] * image_std + image_mean
                viz_imgs.append(rgb_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy())

            H0, W0, H1, W1 = *viz_imgs[0].shape[:2], *viz_imgs[1].shape[:2]
            img0 = np.pad(viz_imgs[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
            img1 = np.pad(viz_imgs[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
            img = np.concatenate((img0, img1), axis=1)
            #pl.figure()
            #pl.imshow(img)
            cmap = pl.get_cmap('jet')
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
            options.max_num_iterations = 2500
            options.success_probability = 0.9999
            options.random_seed = 0 # for reproducibility
            options.final_least_squares = True
            options.threshold_multiplier = 5.0
            options.num_lo_steps = 6
            # squared px thresholds for reprojection error and epipolar error
            reproj_pix_thres = 0.5
            epipolar_pix_thres = 0.5
            epipolar_weight = 0.5
            options.squared_inlier_thresholds = [reproj_pix_thres ** 2, epipolar_pix_thres ** 2]
            # weight when scoring for the two types of errors
            options.data_type_weights = [1.0, epipolar_weight]

            est_config = madpose.EstimatorConfig()
            # if enabled, the input min_depth values are guaranteed to be positive with the estimated depth offsets (shifts), default: True
            est_config.min_depth_constraint = False
            # if disabled, will model the depth with only scale (only applicable to the calibrated camera case)
            est_config.use_shift = False
            # best set to the number of PHYSICAL CPU cores
            est_config.ceres_num_threads = 8

            #intrinsics: [1058.1744780806393, 1058.4470113647467, 675.570437960496, 334.6606098486689]  
            #generate K0 and K1
            K0 = np.array([[1.11765708e+03, 0.00000000e+00, 6.57809477e+02],
                           [0.00000000e+00, 1.22047382e+03, 3.67206231e+02],
                           [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
            K1 = np.array([[1.11765708e+03, 0.00000000e+00, 6.57809477e+02],
                           [0.00000000e+00, 1.22047382e+03, 3.67206231e+02],
                           [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
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
                          x0, x1, 
                          depth0, depth1,
                          min_depth, 
                          K0, K1, options, est_config
                      )
            # rotation and translation of the estimated pose
            R_est, t_est = pose.R(), pose.t()
            # scale and offsets of the affine corrected depth maps
            s_est, o0_est, o1_est = pose.scale, pose.offset0, pose.offset1
            print(">>RANSAC Inliers: ", stats.best_num_inliers)
            print(">>RANSAC Inlier Ratio: ", stats.inlier_ratios)
            
            R_est, t_est = pose.R(), pose.t()
            s_est, o0_est, o1_est = pose.scale, pose.offset0, pose.offset1

            R_est, t_est = pose.R(), pose.t()
            s_est, o0_est, o1_est = pose.scale, pose.offset0, pose.offset1
            print("> Estimated pose: ", pose)
            print("> Estimated scale: ", s_est)
            print("> Estimated offset0: ", o0_est)
            print("> Estimated offset1: ", o1_est)
            print("> Estimated rotation: ", R_est)
            print("> Estimated translation: ", t_est)
            print("==============================")
            #write the estimated rotation and translation into results.txt
            
            #get timestamps
            def get_line_of_timestamp(timestamp):
                #read the timestamps.txt file
                #gt_modified.txt has the ts, x, y, z, qx, qy, qz, qw
                timestamps = np.loadtxt("gt_modified.txt")[:,0]
                #get the line of the timestamp that is closer to the input timestamp
                line = np.argmin(np.abs(timestamps - timestamp))
                timestamps  = np.loadtxt("gt_modified.txt")
                #print  the entire line 
                return timestamps[line]

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


            timestamps = np.loadtxt("timestamps_f.txt")
            print(i)
            print(timestamps[i])
            ts1, Twc1 = pose_line_to_Twc(get_line_of_timestamp(timestamps[i]))
            ts2, Twc2 = pose_line_to_Twc(get_line_of_timestamp(timestamps[i+1]))
            R12, t12 = relative_T12(Twc1, Twc2)
            print("> GT rotation: ", R12)
            print("> GT translation: ", t12)
            with open("results.txt", "a") as f:
                f.write(f"From {time1} to {time2}\n")
                f.write(f"Estimated rotation: {R_est}\n")
                f.write(f"GT rotation: {R12}\n")
                f.write(f"Estimated translation: {t_est}\n")
                f.write(f"GT translation: {t12}\n")
                f.write(f"Number of matches: {num_matches}\n")
                f.write(f"Number of inliers: {stats.best_num_inliers}\n")
                f.write("==============================\n")
            f.close()
            # --- Assume Previous Pose is Identity (Origin) ---
            R_prev = np.eye(3)
            t_prev = np.zeros(3)

            # --- Visualize ---
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # Previous Camera Pose
            draw_camera(ax, R_prev, t_prev, color='orange', label='Before')

            # Estimated (New) Camera Pose
            draw_camera(ax, R_est, t_est, color='blue', label='After')

            # Axes limits
            ax.set_xlim([-0.3, 0.3])
            ax.set_ylim([-0.3, 0.3])
            ax.set_zlim([-0.4, 0.1])

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('Camera Pose Transformation')
            #plt.show()
    


if __name__ == "__main__":
    #timestamps_f.txt has the timestamps of the images
    
        #get the two images
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    '''DEPTH ESTIMATION PART: TWO IMAGES' DEPTH ESTIMATION TAKES PLACE HERE'''
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    parser = argparse.ArgumentParser(description='Inference script', conflict_handler='resolve')
    parser.add_argument("--input", type=str, default = "../mast3r/imgs", help="Path to input image or directory.")
    parser.add_argument("--output", type=str, default = "../mast3r/out", help="Path to output directory.")
    parser.add_argument("--config-file", type=str, default="./configs/eval/vitl.json", help="Path to config file.")
    parser.add_argument("--camera-path", type=str, default=None, help="Path to camera parameters JSON file.")
    parser.add_argument("--save", action="store_true", help="Save outputs as (colorized) PNG.")
    parser.add_argument("--save-ply", action="store_true", help="Save pointcloud as PLY.")
    parser.add_argument("--resolution-level", type=int, default=9, help="Resolution level in [0,10).", choices=list(range(10)))
    parser.add_argument("--interpolation-mode", type=str, default="bilinear", help="Output interpolation.", choices=["nearest", "nearest-exact", "bilinear"])
    args = parser.parse_args()
    print("Parsing done.")
    print("==============================")
    run(args)    
