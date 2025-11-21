import numpy as np 

def bbox_iou(box1, box2):
    """
    Compute Intersection over Union (IoU) between two bounding boxes.

    Parameters
    ----------
    box1 : tuple
        (x_min, y_min, x_max, y_max)
    box2 : tuple
        (x_min, y_min, x_max, y_max)

    Returns
    -------
    float
        IoU value in [0, 1].
        Returns 0 if boxes do not overlap.

    Notes
    -----
    - Boxes are inclusive pixel coordinates.
    - Works for integer or float coordinates.
    """
    if box1 is None or box2 is None:
        return 0.0

    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Intersection coords
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    # If no overlap
    if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
        return 0.0

    # Areas
    inter_area = (inter_xmax - inter_xmin + 1) * (inter_ymax - inter_ymin + 1)
    area1 = (x1_max - x1_min + 1) * (y1_max - y1_min + 1)
    area2 = (x2_max - x2_min + 1) * (y2_max - y2_min + 1)

    union_area = area1 + area2 - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area

def mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Calculate the Intersection over Union (IoU) between two binary masks.

    IoU = (intersection area) /     response = predictor.handle_request(
        request=dict(
            type="remove_object",
            session_id=session_id,
            obj_id=obj_id,
        )
    )(union area)

    Parameters
    ----------
    mask1 : np.ndarray
        A 2D binary mask (values 0 or 1 / False or True).
    mask2 : np.ndarray
        A 2D binary mask of the same shape.

    Returns
    -------
    float
        IoU score in the range [0.0, 1.0].
        Returns 0.0 if both masks are empty or union is zero.

    Raises
    ------
    ValueError
        If the input masks do not have the same shape.

    Notes
    -----
    - The function assumes masks are binary. If masks contain other values,
      they will be treated as boolean (non-zero = True).
    - IoU is commonly used for segmentation evaluation.
    """
    if mask1.shape != mask2.shape:
        raise ValueError("Masks must have the same shape.")

    # Convert to boolean
    m1 = mask1.astype(bool)
    m2 = mask2.astype(bool)

    # Compute intersection and union
    intersection = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()

    # Avoid division by zero
    if union == 0:
        return 0.0

    return intersection / union

def mask_to_bbox(mask: np.ndarray):
    """
    Convert a 2D binary mask into a bounding box.

    Parameters
    ----------
    mask : np.ndarray
        A 2D binary mask (values 0/1 or False/True).

    Returns
    -------
    tuple or None
        (x_min, y_min, x_max, y_max) if mask contains at least one pixel.
        None if the mask is empty.

    Notes
    -----
    - Coordinates follow image convention: (y, x) order in indexing,
      but output is returned as (x_min, y_min, x_max, y_max).
    - x_max and y_max are inclusive pixel indices.
    """
    mask = mask.astype(bool)

    if not mask.any():
        return None

    # Get coordinates of non-zero pixels
    ys, xs = np.where(mask)

    x_min = xs.min()
    x_max = xs.max()
    y_min = ys.min()
    y_max = ys.max()

    return np.array([x_min, y_min, x_max, y_max])

def sample_points_from_mask(mask: np.ndarray, num_points: int,
                            replace: bool = False,
                            return_xy: bool = False):
    """
    Randomly sample points from a binary mask.

    Parameters
    ----------
    mask : np.ndarray
        2D binary mask (values 0/1 or False/True).
    num_points : int
        Number of points to sample.
    replace : bool, optional
        If True, sample with replacement. Default is False.
    return_xy : bool, optional
        If True, return (x, y) coordinates instead of (y, x).

    Returns
    -------
    np.ndarray
        Array of shape (N, 2) with sampled pixel coordinates.
        Columns: (y, x) by default, or (x, y) if return_xy=True.

    Raises
    ------
    ValueError
        If mask has no active pixels.
        If sampling without replacement and num_points > available pixels.
    """
    # Convert to boolean mask
    mask = mask.astype(bool)

    # Find foreground pixel coordinates
    ys, xs = np.where(mask)
    coords = np.stack([ys, xs], axis=1)

    if len(coords) == 0:
        raise ValueError("Mask is empty — no points to sample.")

    if not replace and num_points > len(coords):
        raise ValueError(
            f"Cannot sample {num_points} points without replacement; "
            f"mask has only {len(coords)} valid pixels."
        )

    # Sample indices
    idx = np.random.choice(len(coords), size=num_points, replace=replace)
    sampled = coords[idx]

    if return_xy:
        # swap columns (y,x) → (x,y)
        sampled = sampled[:, [1, 0]]

    return sampled

import cv2
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import subprocess
from skimage.color import lab2rgb, rgb2lab
from sklearn.cluster import KMeans

def generate_colors(n_colors=256, n_samples=5000):
    # Step 1: Random RGB samples
    np.random.seed(42)
    rgb = np.random.rand(n_samples, 3)
    # Step 2: Convert to LAB for perceptual uniformity
    # print(f"Converting {n_samples} RGB samples to LAB color space...")
    lab = rgb2lab(rgb.reshape(1, -1, 3)).reshape(-1, 3)
    # print("Conversion to LAB complete.")
    # Step 3: k-means clustering in LAB
    kmeans = KMeans(n_clusters=n_colors, n_init=10)
    # print(f"Fitting KMeans with {n_colors} clusters on {n_samples} samples...")
    kmeans.fit(lab)
    # print("KMeans fitting complete.")
    centers_lab = kmeans.cluster_centers_
    # Step 4: Convert LAB back to RGB
    colors_rgb = lab2rgb(centers_lab.reshape(1, -1, 3)).reshape(-1, 3)
    colors_rgb = np.clip(colors_rgb, 0, 1)
    return colors_rgb


COLORS = generate_colors(n_colors=128, n_samples=5000)


def load_frame(frame):
    if isinstance(frame, np.ndarray):
        img = frame
    elif isinstance(frame, Image.Image):
        img = np.array(frame)
    elif isinstance(frame, str) and os.path.isfile(frame):
        img = plt.imread(frame)
    else:
        raise ValueError(f"Invalid video frame type: {type(frame)=}")
    return img

def render_masklet_frame(img, outputs, frame_idx=None, alpha=0.5):
    """
    Overlays masklets and bounding boxes on a single image frame.
    Args:
        img: np.ndarray, shape (H, W, 3), uint8 or float32 in [0,255] or [0,1]
        outputs: dict with keys: out_boxes_xywh, out_probs, out_obj_ids, out_binary_masks
        frame_idx: int or None, for overlaying frame index text
        alpha: float, mask overlay alpha
    Returns:
        overlay: np.ndarray, shape (H, W, 3), uint8
    """
    if img.dtype == np.float32 or img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    img = img[..., :3]  # drop alpha if present
    height, width = img.shape[:2]
    overlay = img.copy()

    for i in range(len(outputs["scores"])):
        obj_id = outputs["object_ids"][i]
        color = COLORS[obj_id % len(COLORS)]
        color255 = (color * 255).astype(np.uint8)
        mask = outputs["masks"][i]
        if hasattr(mask, "detach"):
            mask = mask.detach().cpu().numpy()
        if mask.shape != img.shape[:2]:
            mask = cv2.resize(
                mask.astype(np.float32),
                (img.shape[1], img.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        mask_bool = mask > 0.5
        for c in range(3):
            overlay[..., c][mask_bool] = (
                alpha * color255[c] + (1 - alpha) * overlay[..., c][mask_bool]
            ).astype(np.uint8)

    # Draw bounding boxes and text
    for i in range(len(outputs["scores"])):
        box_xyxy = outputs["boxes"][i]
        obj_id = outputs["object_ids"][i]
        prob = outputs["scores"][i]
        color = COLORS[obj_id % len(COLORS)]
        color255 = tuple(int(x * 255) for x in color)
        x1, y1, x2, y2 = [int(v) for v in box_xyxy.to("cpu").tolist()]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color255, 2)
        if prob is not None:
            label = f"id={obj_id}, p={prob:.2f}"
        else:
            label = f"id={obj_id}"
        cv2.putText(
            overlay,
            label,
            (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color255,
            1,
            cv2.LINE_AA,
        )

    # Overlay frame index at the top-left corner
    if frame_idx is not None:
        cv2.putText(
            overlay,
            f"Frame {frame_idx}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return overlay


def save_masklet_video(video_frames, outputs, out_path, alpha=0.5, fps=10):
    # Each outputs dict has keys: "out_boxes_xywh", "out_probs", "out_obj_ids", "out_binary_masks"
    # video_frames: list of video frame data, same length as outputs_list

    # Read first frame to get size
    first_img = load_frame(video_frames[0])
    height, width = first_img.shape[:2]
    if first_img.dtype == np.float32 or first_img.max() <= 1.0:
        first_img = (first_img * 255).astype(np.uint8)
    # Use 'mp4v' for best compatibility with VSCode playback (.mp4 files)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter("temp.mp4", fourcc, fps, (width, height))

    outputs_list = [
        (video_frames[frame_idx], frame_idx, outputs[frame_idx])
        for frame_idx in sorted(outputs.keys())
    ]

    for frame, frame_idx, frame_outputs in tqdm(outputs_list):
        img = load_frame(frame)
        overlay = render_masklet_frame(
            img, frame_outputs, frame_idx=frame_idx, alpha=alpha
        )
        writer.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    writer.release()

    # Re-encode the video for VSCode compatibility using ffmpeg
    subprocess.run(["ffmpeg", "-y", "-i", "temp.mp4", out_path])
    print(f"Re-encoded video saved to {out_path}")

    os.remove("temp.mp4")  # Clean up temporary file
