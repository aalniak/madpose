# monocular_BA_pipeline.py
"""
Depth-aided two-frame pose pipeline + optional bundle-adjustment (BA).
Key runtime flags
─────────────────
--input <img_folder>      folder with monocular images
--timestamps <txt>        line-aligned timestamps (same order as images)
--debug N                 process first N pairs then exit
--force_cpu               disable CUDA entirely (sets CUDA_VISIBLE_DEVICES="")
"""

import sys
## Add all paths before any imports
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
import cv2
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rscipy
from PIL import Image

sys.path.append('/home/arda/OGAM/UniK3D')
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

# ------------------------------------------------------------------
# Early device selection that survives broken CUDA installs
# ------------------------------------------------------------------

def select_device(force_cpu: bool):
    if force_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # torch will think GPU absent
        return "cpu"
    try:
        # Set CUDA_VISIBLE_DEVICES before importing torch
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Only use first GPU
        #import torch
        if not torch.cuda.is_available():
            print("[WARN] CUDA not available - falling back to CPU")
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            return "cpu"
        try:
            # Test CUDA device
            test_tensor = torch.zeros(1).cuda()
            del test_tensor
            return "cuda"
        except RuntimeError as e:
            print(f"[WARN] CUDA device test failed - falling back to CPU ({e})")
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            return "cpu"
    except Exception as e:
        print(f"[WARN] CUDA init failed – falling back to CPU ({e})")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return "cpu"

DEVICE = select_device(force_cpu=False)
print(f"[INFO] Running on {DEVICE.upper()}")

#INTRINSICS FOR M3ED SPOT SEQUENCES
K_INTR = np.array([[1.11765708e3, 0.,           6.57809477e2],
                   [0.,           1.22047382e3, 3.67206231e2],
                   [0.,           0.,           1.]])

def _pack(Rws, tws, pts3d):
    cams = [np.hstack([Rscipy.from_matrix(Rw).as_rotvec(), tw]) for Rw, tw in zip(Rws, tws)]
    return np.hstack([*cams, pts3d.ravel()])

def _unpack(x, n_cams, n_pts):
    cams = x[: n_cams * 6].reshape(n_cams, 6)
    Rws, tws = [], []
    for cam in cams:
        rotvec = cam[:3]  # First 3 elements are rotation vector
        t = cam[3:]      # Last 3 elements are translation
        Rws.append(Rscipy.from_rotvec(rotvec).as_matrix())
        tws.append(t)
    pts3d = x[n_cams * 6:].reshape(n_pts, 3)
    return Rws, tws, pts3d

def _reproj_res_torch(params, n_cams, n_pts, tracks, K):
    # Unpack parameters
    cams = params[:n_cams * 6].view(n_cams, 6)
    pts3d = params[n_cams * 6:].view(n_pts, 3)
    
    # Extract camera parameters
    Rws = []
    tws = []
    for i in range(n_cams):
        rotvec = cams[i, :3]
        t = cams[i, 3:]
        Rw = torch.matrix_exp(torch.tensor([[0, -rotvec[2], rotvec[1]],
                                          [rotvec[2], 0, -rotvec[0]],
                                          [-rotvec[1], rotvec[0], 0]], device=params.device))
        Rws.append(Rw)
        tws.append(t)
    
    # Camera intrinsics
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    
    # Compute reprojection errors
    errors = []
    for j, Xw in enumerate(pts3d):
        for kf, u, v in tracks[j]:
            Xc = Rws[kf].T @ (Xw - tws[kf])
            if Xc[2] <= 1e-4:
                continue
            x = fx * Xc[0] / Xc[2] + cx
            y = fy * Xc[1] / Xc[2] + cy
            errors.extend([x - u, y - v])
    
    return torch.stack(errors) if errors else torch.zeros(1, device=params.device)

def _seed_tracks(R0, t0, R1, t1, mk0, mk1, depth0):
    # Convert depth0 to numpy if it's a tensor
    if torch.is_tensor(depth0):
        depth0 = depth0.cpu().numpy()
    
    Xcs = np.linalg.inv(K_INTR) @ np.vstack([mk0.T, np.ones(len(mk0))])
    Xws = (R0 @ (depth0 * Xcs)).T + t0
    pts, tracks = [], []
    for idx, Xw in enumerate(Xws): 
        pts.append(Xw)
        tracks.append([(0, *mk0[idx]), (1, *mk1[idx])])
    return np.asarray(pts), tracks

def ba_two_frame(R, t, mk0, mk1, d0, debug=False):
    print("BA started.")
    pts, trk = _seed_tracks(np.eye(3), np.zeros(3), R, t, mk0, mk1, d0)
    print("Seed tracks done.")
    if len(trk) < 8:
        return R, t
    
    # Convert to PyTorch tensors
    device = torch.device('cuda')
    x0 = torch.tensor(_pack([np.eye(3), R.copy()], [np.zeros(3), t.copy()], pts), 
                     device=device, dtype=torch.float32, requires_grad=True)
    K_torch = torch.tensor(K_INTR, device=device, dtype=torch.float32)
    
    print("Optimization started.")
    t0 = time.perf_counter()
    
    # Optimization parameters
    lr = 0.01
    max_iter = 21
    optimizer = torch.optim.Adam([x0], lr=lr)
    
    try:
        for i in range(max_iter):
            optimizer.zero_grad()
            errors = _reproj_res_torch(x0, 2, len(pts), trk, K_torch)
            loss = torch.mean(torch.abs(errors))  # L1 loss
            loss.backward()
            optimizer.step()
            
            if i % 10 == 0:
                print(f"Iteration {i}, Loss: {loss.item():.6f}")
            
            if loss.item() < 1e-4:  # Early stopping
                break
                
        print("Optimization done.")
    except Exception as e:
        print(f"Optimization failed: {e}")
        return R, t
    
    if debug:
        print(f"BA completed in {time.perf_counter()-t0:.3f}s")
    
    # Convert back to numpy
    x0_np = x0.detach().cpu().numpy()
    Rws, tws, _ = _unpack(x0_np, 2, len(pts))
    return Rws[1], tws[1]

# ---------------- wrappers for depth & matches ------------------- #

def infer_depth(model, img):
    arr = np.array(Image.open(img));  ten = torch.as_tensor(arr).permute(2,0,1)
    return model.infer(rgb=ten, camera=None, normalize=True, rays=None)['depth'].cpu().squeeze()

def match_pts(model, img0, img1, device, max_m=7000):
    ims = load_images([img0, img1], size=1280)
    out = inference([tuple(ims)], model, device, batch_size=1, verbose=False)
    d1, d2 = out['pred1']['desc'].squeeze(0), out['pred2']['desc'].squeeze(0)
    mk0, mk1 = fast_reciprocal_NNs(d1, d2, subsample_or_initxy1=8, device=device, dist='dot')
    if len(mk0) > max_m:
        sel = np.random.choice(len(mk0), max_m, replace=False)
        mk0, mk1 = mk0[sel], mk1[sel]
    return mk0, mk1

# ---------------- main pipeline ---------------------------------- #

def run_unik3d_stage(args, img_path0, img_path1):
    """Stage 1: Depth estimation using UniK3D"""
    print("\n=== Stage 1: Depth Estimation (UniK3D) ===")
    version = args.config_file.split("/")[-1].split(".")[0]
    name = f"unik3d-{version}"
    model = UniK3D.from_pretrained(f"lpiccinelli/{name}")
    print("> UniK3D loaded.")
    model.resolution_level = args.resolution_level
    model.interpolation_mode = args.interpolation_mode

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    # Process first image
    args_single = argparse.Namespace(**vars(args))
    args_single.input = img_path0
    depth_map0 = infer_depth(model, img_path0)
    depth_map0 = depth_map0.cpu().squeeze()
    print("Depth map 0 mean: ", depth_map0.mean())
    
    # Process second image
    args_single.input = img_path1
    depth_map1 = infer_depth(model, img_path1)
    depth_map1 = depth_map1.cpu().squeeze()
    print("Depth map 1 mean: ", depth_map1.mean())
    
    print("> Depth maps inferred.")
    del model 
    gc.collect()
    torch.cuda.empty_cache()
    print("> UniK3D deleted to free VRAM.")
    print("=== Depth Estimation Complete ===\n")
    
    return depth_map0, depth_map1

def run_mast3r_stage(args, img_path0, img_path1):
    """Stage 2: Point Matching using MASt3R"""
    print("\n=== Stage 2: Point Matching (MASt3R) ===")
    device = 'cuda'
    model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    print("> MASt3R loaded.")
    
    mast3r_model = AsymmetricMASt3R.from_pretrained(model_name).to(device)
    
    # Load and process images
    images = load_images([img_path0, img_path1], size=1280)
    output = inference([tuple(images)], mast3r_model, device, batch_size=1, verbose=False)

    # Extract predictions
    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']
    desc1, desc2 = pred1['desc'].squeeze(0).detach(), pred2['desc'].squeeze(0).detach()
    
    # Get confidence scores
    conf1 = pred1['conf'].squeeze(0).detach() if 'conf' in pred1 else None
    conf2 = pred2['conf'].squeeze(0).detach() if 'conf' in pred2 else None

    # Find matches
    matches_im0, matches_im1 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=8,
                                                   device=device, dist='dot', block_size=2**13)
    num_matches = matches_im0.shape[0]
    print("Number of matches from Mast3R point matching: ", num_matches)
    
    # Get confidence scores for matches
    if conf1 is not None and conf2 is not None:
        # Get confidence scores for matched points
        conf_scores = torch.min(conf1[matches_im0[:, 1], matches_im0[:, 0]], 
                              conf2[matches_im1[:, 1], matches_im1[:, 0]])
        
        # Sort by confidence
        sorted_indices = torch.argsort(conf_scores, descending=True)
        matches_im0 = matches_im0[sorted_indices]
        matches_im1 = matches_im1[sorted_indices]
        
        # Take top 2000 matches for MADPose
        max_matches_madpose = 2000
        matches_im0_madpose = matches_im0[:max_matches_madpose]
        matches_im1_madpose = matches_im1[:max_matches_madpose]
        print(f"Selected top {max_matches_madpose} matches for MADPose based on confidence")
        
        # Take top 200 matches for bundle adjustment
        max_matches_ba = 200
        matches_im0_ba = matches_im0[:max_matches_ba]
        matches_im1_ba = matches_im1[:max_matches_ba]
        print(f"Selected top {max_matches_ba} matches for bundle adjustment based on confidence")
    else:
        # If no confidence scores, randomly sample
        max_matches_madpose = 2000
        max_matches_ba = 200
        if num_matches > max_matches_madpose:
            indices = np.random.choice(num_matches, max_matches_madpose, replace=False)
            matches_im0_madpose = matches_im0[indices]
            matches_im1_madpose = matches_im1[indices]
            print(f"Randomly selected {max_matches_madpose} matches for MADPose (no confidence scores available)")
            
            indices = np.random.choice(num_matches, max_matches_ba, replace=False)
            matches_im0_ba = matches_im0[indices]
            matches_im1_ba = matches_im1[indices]
            print(f"Randomly selected {max_matches_ba} matches for bundle adjustment (no confidence scores available)")
    
    # Filter valid matches for MADPose
    H0, W0 = view1['true_shape'][0]
    valid_matches_im0 = (matches_im0_madpose[:, 0] >= 3) & (matches_im0_madpose[:, 0] < int(W0) - 3) & (
        matches_im0_madpose[:, 1] >= 3) & (matches_im0_madpose[:, 1] < int(H0) - 3)

    H1, W1 = view2['true_shape'][0]
    valid_matches_im1 = (matches_im1_madpose[:, 0] >= 3) & (matches_im1_madpose[:, 0] < int(W1) - 3) & (
        matches_im1_madpose[:, 1] >= 3) & (matches_im1_madpose[:, 1] < int(H1) - 3)

    valid_matches = valid_matches_im0 & valid_matches_im1
    matches_im0_madpose, matches_im1_madpose = matches_im0_madpose[valid_matches], matches_im1_madpose[valid_matches]

    print("> Point matching done.")
    del mast3r_model
    gc.collect()
    torch.cuda.empty_cache()
    print("> MASt3R deleted to free VRAM.")
    print("=== Point Matching Complete ===\n")
    
    return matches_im0_madpose, matches_im1_madpose, matches_im0_ba, matches_im1_ba

def run_madpose_stage(args, matches_im0, matches_im1, matches_im0_ba, matches_im1_ba, depth_map0, depth_map1, pair_index):
    """Stage 3: Pose Estimation using MADPose"""
    print("\n=== Stage 3: Pose Estimation (MADPose) ===")
    
    # Prepare data for MADPose
    mkpts0 = matches_im0
    mkpts1 = matches_im1
    depth_map0_cpu = depth_map0.cpu().squeeze()
    depth_map1_cpu = depth_map1.cpu().squeeze()
    
    # Extract x and y coordinates
    x0, y0 = mkpts0[:, 0], mkpts0[:, 1]
    x1, y1 = mkpts1[:, 0], mkpts1[:, 1]
    
    # Get depth values using integer coordinates
    depth0 = depth_map0_cpu[y0.astype(np.int64), x0.astype(np.int64)]
    depth1 = depth_map1_cpu[y1.astype(np.int64), x1.astype(np.int64)]

    # Configure MADPose options
    options = madpose.HybridLORansacOptions()
    options.min_num_iterations = 100
    options.max_num_iterations = 2500
    options.success_probability = 0.9999
    options.random_seed = 0
    options.final_least_squares = True
    options.threshold_multiplier = 5.0
    options.num_lo_steps = 6
    reproj_pix_thres = 0.5
    epipolar_pix_thres = 0.5
    epipolar_weight = 0.5
    options.squared_inlier_thresholds = [reproj_pix_thres ** 2, epipolar_pix_thres ** 2]
    options.data_type_weights = [1.0, epipolar_weight]

    est_config = madpose.EstimatorConfig()
    est_config.min_depth_constraint = False
    est_config.use_shift = False
    est_config.ceres_num_threads = 8

    # Format inputs for MADPose
    x0_madpose = [np.array([[x], [y]], dtype=np.float64) for x, y in zip(x0, y0)]
    x1_madpose = [np.array([[x], [y]], dtype=np.float64) for x, y in zip(x1, y1)]
    depth0 = [float(d) for d in depth0]
    depth1 = [float(d) for d in depth1]
    min_depth = np.array([[depth_map0_cpu.min()], [depth_map1_cpu.min()]], dtype=np.float64)

    # Estimate pose
    pose, stats = madpose.HybridEstimatePoseScaleOffset(
                  x0_madpose, x1_madpose, 
                  depth0, depth1,
                  min_depth, 
                  K_INTR, K_INTR, options, est_config
              )
    
    R_est, t_est = pose.R(), pose.t()
    
    # Optional bundle adjustment - only after pair 12
    if not args.no_ba and pair_index >= 12:
        # Use top 200 high-confidence matches for bundle adjustment
        print(f"Using top {len(matches_im0_ba)} high-confidence matches for bundle adjustment")
        
        # Get depths for BA matches
        depth0_ba = depth_map0_cpu[matches_im0_ba[:, 1].astype(np.int64), 
                                 matches_im0_ba[:, 0].astype(np.int64)]
        
        R_est, t_est = ba_two_frame(R_est, t_est, matches_im0_ba, matches_im1_ba, 
                                  depth0_ba, args.debug)
    elif not args.no_ba:
        print(f"Skipping bundle adjustment for pair {pair_index} (before pair 12)")
    
    print("> Pose estimation complete.")
    print(f"Number of inliers: {stats.best_num_inliers}")
    print("=== Pose Estimation Complete ===\n")
    
    return R_est, t_est, stats

def get_line_of_timestamp(timestamp):
    #read the timestamps.txt file
    #gt_modified.txt has the ts, x, y, z, qx, qy, qz, qw
    timestamps = np.loadtxt("gt_modified.txt")[:,0]
    #get the line of the timestamp that is closer to the input timestamp
    line = np.argmin(np.abs(timestamps - timestamp))
    timestamps = np.loadtxt("gt_modified.txt")
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

def run(args):
    """Main pipeline that runs all stages sequentially"""
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir:  # Only create directory if path contains a directory
        os.makedirs(output_dir, exist_ok=True)
    
    # Load timestamps
    timestamps = np.loadtxt(args.timestamps)
    print(f"Loaded {len(timestamps)} timestamps")
    
    # Find the starting index if start_time is specified
    start_idx = 0
    if args.start_time is not None:
        # Find the closest timestamp to start_time
        start_idx = np.argmin(np.abs(timestamps - args.start_time))
        print(f"Starting from timestamp {timestamps[start_idx]:.6f} (closest to {args.start_time:.6f})")
    
    # Process each consecutive pair
    for i in range(start_idx, len(timestamps)-1):
        if args.debug and i >= args.debug:
            break
            
        time1 = timestamps[i]
        time2 = timestamps[i+1]
        
        print(f"\nProcessing pair {i+1}/{len(timestamps)-1}")
        print(f"Timestamps: {time1:.6f} -> {time2:.6f}")
        
        img_path0 = f"{args.input}/{int(round(time1*10))}0000000.png"
        img_path1 = f"{args.input}/{int(round(time2*10))}0000000.png"
        
        if not os.path.exists(img_path0) or not os.path.exists(img_path1):
            print(f"Skipping pair {i+1}: Image files not found")
            continue
        
        try:
            # Stage 1: Depth Estimation
            depth_map0, depth_map1 = run_unik3d_stage(args, img_path0, img_path1)
            
            # Stage 2: Point Matching
            matches_im0, matches_im1, matches_im0_ba, matches_im1_ba = run_mast3r_stage(args, img_path0, img_path1)
            
            # Stage 3: Pose Estimation
            R_est, t_est, stats = run_madpose_stage(args, matches_im0, matches_im1, matches_im0_ba, matches_im1_ba, 
                                                  depth_map0, depth_map1, i+1)
            
            # Get ground truth poses
            ts1, Twc1 = pose_line_to_Twc(get_line_of_timestamp(time1))
            ts2, Twc2 = pose_line_to_Twc(get_line_of_timestamp(time2))
            R_gt, t_gt = relative_T12(Twc2, Twc1)
            
            # Save results in the requested format
            with open(args.output, 'a') as log:
                log.write(f"From {time1:.1f} to {time2:.1f}\n")
                log.write("Estimated rotation:\n")
                log.write(f"{R_est}\n")
                log.write("GT rotation:\n")
                log.write(f"{R_gt}\n")
                log.write("Estimated translation:\n")
                log.write(f"{t_est}\n")
                log.write("GT translation:\n")
                log.write(f"{t_gt}\n")
                log.write(f"Number of matches: {len(matches_im0)}\n")
                log.write(f"Number of inliers: {stats.best_num_inliers}\n")
                log.write("==============================\n\n")
            
            print(f"Completed processing pair {i+1}")
            print("==============================")
            
        except Exception as e:
            print(f"Error processing pair {i+1}: {str(e)}")
            continue
        
        # Clear GPU memory after each pair
        torch.cuda.empty_cache()
        gc.collect()

# ---------------- CLI ------------------------------------------- #

# -------------------- CLI ------------------------------------------ #
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default="../mast3r/asd", help='folder with images')
    ap.add_argument('--timestamps', default='timestamps_f.txt', help='timestamps txt')
    ap.add_argument('--config-file', default='./configs/eval/vitl.json')
    ap.add_argument('--output', default='results_BA.txt')
    ap.add_argument('--debug', type=int, default=0, help='0=full, N=run N pairs then exit')
    ap.add_argument('--no_ba', default=True, help='skip further BA stage entirely')
    ap.add_argument('--resolution_level', type=int, default=0, help='resolution level for UniK3D')
    ap.add_argument('--interpolation_mode', type=str, default='bilinear', help='interpolation mode for UniK3D')
    ap.add_argument('--start_time', type=float, default=4.4, help='start processing from this timestamp')
    args = ap.parse_args()
    run(args)
