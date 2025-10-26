import streamlit as st
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
import matplotlib.pyplot as plt
from io import BytesIO
import json

# Set page config
st.set_page_config(page_title="SAM2 Segmentation Labeling", layout="wide")

# Initialize session state
if "current_image_idx" not in st.session_state:
    st.session_state.current_image_idx = 0
if "clicks" not in st.session_state:
    st.session_state.clicks = (
        []
    )  # List of tuples: [(x, y, label), ...] where label is 1 (positive) or 0 (negative)
if "click_mode" not in st.session_state:
    st.session_state.click_mode = 1  # 1 for positive, 0 for negative
if "image_list" not in st.session_state:
    st.session_state.image_list = []
if "model" not in st.session_state:
    st.session_state.model = None
if "predictor" not in st.session_state:
    st.session_state.predictor = None
if "current_mask" not in st.session_state:
    st.session_state.current_mask = None
if "labeled_images" not in st.session_state:
    st.session_state.labeled_images = set()

# Configuration
base_path = "./automated_underwater_area_estimation/"
IMAGES_DIR = Path(base_path + "data_preprocessed/IBF/images")
MASKS_DIR = Path(base_path + "data_preprocessed/IBF/masks")
CLICKS_DIR = Path(base_path + "data_preprocessed/IBF/clicks")
CHECKPOINT_PATH = base_path + "label_studio//aquasam_weights.pth"  # AquaSAM checkpoint
SAM_VARIANT = "vit_b"  # AquaSAM is ViT-B
TARGET_SIZE = 1024
SAVE_SIZE = (4000, 3000)  # Width x Height for saved masks

# Create masks and clicks directories if they don't exist
MASKS_DIR.mkdir(parents=True, exist_ok=True)
CLICKS_DIR.mkdir(parents=True, exist_ok=True)


@st.cache_resource
def load_model(
    checkpoint_path: str = CHECKPOINT_PATH, sam_variant: str = SAM_VARIANT
) -> Tuple[Any, Any, str]:
    """
    Load AquaSAM model with caching.

    Initializes the Segment Anything Model (SAM) with AquaSAM weights
    fine-tuned for underwater imagery. Uses Streamlit caching for efficiency.

    Args:
        checkpoint_path: Path to AquaSAM checkpoint file
        sam_variant: SAM model variant (e.g., 'vit_b')

    Returns:
        Tuple of (sam_model, predictor, device_name)
    """
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    # Build SAM and load AquaSAM weights (fine-tuned on underwater data)
    sam = sam_model_registry[sam_variant](checkpoint=checkpoint_path)
    sam.to(device=device)
    sam.eval()

    predictor = SamPredictor(sam)
    return sam, predictor, device


def resize_image_proportional(
    image: Image.Image, target_size: int = 1024
) -> Image.Image:
    """
    Resize image so the smaller side equals target_size, maintaining aspect ratio.

    Args:
        image: Input PIL Image
        target_size: Target size for the smaller dimension (default 1024)

    Returns:
        Resized PIL Image
    """
    width, height = image.size

    # Determine which side is smaller
    if width < height:
        # Width is smaller, scale it to target_size
        new_width = target_size
        new_height = int(height * (target_size / width))
    else:
        # Height is smaller, scale it to target_size
        new_height = target_size
        new_width = int(width * (target_size / height))

    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    return resized_image


def upscale_mask_to_original(mask: torch.Tensor, target_size: tuple) -> torch.Tensor:
    """
    Upscale mask tensor to target size (width, height).

    Args:
        mask: Input mask tensor (H, W)
        target_size: Target size as (width, height)

    Returns:
        Upscaled mask tensor
    """
    target_width, target_height = target_size

    # Add batch and channel dimensions for interpolation
    mask_4d = mask.unsqueeze(0).unsqueeze(0).float()

    # Interpolate to target size (note: interpolate expects (height, width))
    upscaled = torch.nn.functional.interpolate(
        mask_4d, size=(target_height, target_width), mode="nearest"
    )

    # Remove batch and channel dimensions and convert back to bool
    upscaled_mask = upscaled.squeeze().bool()

    return upscaled_mask


def get_unlabeled_images() -> Tuple[List[Path], set]:
    """
    Get lists of unlabeled and labeled images.

    Scans the images directory and checks which images already have
    corresponding mask files in the masks directory.

    Returns:
        Tuple of (unlabeled_images_list, labeled_images_set)
    """
    all_images = sorted(
        [f for f in IMAGES_DIR.glob("*.JPG") if f.is_file()]
        + [f for f in IMAGES_DIR.glob("*.jpg") if f.is_file()]
    )

    # Check which images already have masks
    labeled = set()
    for img_path in all_images:
        mask_path = MASKS_DIR / f"{img_path.stem}.pt"
        if mask_path.exists():
            labeled.add(img_path)

    unlabeled = [img for img in all_images if img not in labeled]
    return unlabeled, labeled


def segment_object_from_clicks(
    image: Image.Image, clicks: list, predictor: Any, device: str
) -> Optional[torch.Tensor]:
    """
    Segment an object from an image using AquaSAM with multiple click points.

    Uses the Segment Anything Model to generate a segmentation mask based on
    user-provided positive and negative click points.

    Args:
        image: PIL Image to segment
        clicks: List of (x, y, label) tuples where label is 1 (positive) or 0 (negative)
        predictor: SAM predictor instance
        device: Device name (e.g., 'cuda', 'cpu')

    Returns:
        Boolean tensor mask of shape (H, W), or None if no clicks provided
    """
    if len(clicks) == 0:
        return None

    # Convert PIL image to numpy array
    img_np = np.array(image)  # H x W x 3, uint8

    # Set image for predictor
    predictor.set_image(img_np)

    # Separate points and labels from clicks
    # clicks is now a list of tuples: [(x, y, label), ...]
    pts = np.array(
        [(click[0], click[1]) for click in clicks], dtype=np.float32
    )  # (K, 2) in (x, y)
    lbl = np.array(
        [click[2] for click in clicks], dtype=np.int32
    )  # labels: 1 for positive, 0 for negative

    # Predict
    masks, scores, logits = predictor.predict(
        point_coords=pts, point_labels=lbl, multimask_output=False  # Get one best mask
    )

    # masks -> (N, H, W) boolean; we asked for 1 variant, so N==1
    mask_bool = masks[0]
    mask_tensor = torch.from_numpy(mask_bool)  # (H, W), True/False

    return mask_tensor


def visualize_segmentation(image, mask, clicks):
    """Create visualization with mask overlay and click points"""
    # Convert PIL image to numpy array
    img_array = np.array(image)
    img_height, img_width = img_array.shape[:2]

    # Create figure with exact pixel dimensions
    dpi = 100
    fig = plt.figure(figsize=(img_width / dpi, img_height / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])  # Fill entire figure
    ax.imshow(image)

    if mask is not None:
        ax.imshow(mask, alpha=0.5, cmap="Blues")

    if len(clicks) > 0:
        # Separate positive and negative clicks
        positive_clicks = [(click[0], click[1]) for click in clicks if click[2] == 1]
        negative_clicks = [(click[0], click[1]) for click in clicks if click[2] == 0]

        # Scale marker size based on image dimensions
        marker_size = max(50, (img_width + img_height) / 30)

        # Plot positive clicks in green
        if positive_clicks:
            positive_array = np.array(positive_clicks)
            ax.scatter(
                positive_array[:, 0],
                positive_array[:, 1],
                c="green",
                s=marker_size,
                marker="o",
                edgecolors="white",
                linewidths=3,
            )

        # Plot negative clicks in red
        if negative_clicks:
            negative_array = np.array(negative_clicks)
            ax.scatter(
                negative_array[:, 0],
                negative_array[:, 1],
                c="red",
                s=marker_size,
                marker="o",
                edgecolors="white",
                linewidths=3,
            )

    ax.axis("off")
    ax.set_xlim(0, img_width)
    ax.set_ylim(img_height, 0)  # Invert y-axis to match image coordinates

    # Convert plot to image
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    plt.close()

    result_img = Image.open(buf)
    # Ensure output is exactly the same size as input
    if result_img.size != image.size:
        result_img = result_img.resize(image.size, Image.LANCZOS)

    return result_img


def save_segmentation(image_path, mask, clicks, working_size, save_size):
    """
    Save mask as binary tensor upscaled to save_size and clicks as JSON.

    Args:
        image_path: Path to the image file
        mask: Mask tensor at working resolution
        clicks: Click coordinates with labels at working resolution [(x, y, label), ...]
        working_size: Size of the working image (width, height)
        save_size: Target size for saved mask (width, height) - e.g., (4000, 3000)
    """
    # Upscale mask to save size
    upscaled_mask = upscale_mask_to_original(mask, save_size)

    # Save mask as binary tensor
    mask_path = MASKS_DIR / f"{image_path.stem}_mask.pt"
    torch.save(upscaled_mask, mask_path)

    # Calculate scale factors to convert clicks to save size
    scale_x = save_size[0] / working_size[0]
    scale_y = save_size[1] / working_size[1]

    # Scale clicks to save size coordinates (preserve labels)
    scaled_clicks = [
        [int(x * scale_x), int(y * scale_y), label] for x, y, label in clicks
    ]

    # Save metadata
    metadata_path = CLICKS_DIR / f"{image_path.stem}_metadata.json"
    metadata = {
        "clicks_working": clicks,  # Clicks at working resolution with labels
        "clicks_original": scaled_clicks,  # Clicks scaled to save resolution with labels
        "working_size": list(working_size),
        "save_size": list(save_size),
        "mask_size": list(upscaled_mask.shape),
        "image_name": image_path.name,
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return mask_path


# Load model
with st.spinner("Loading AquaSAM model..."):
    if st.session_state.model is None:
        model, predictor, device = load_model()
        st.session_state.model = model
        st.session_state.predictor = predictor
        st.session_state.device = device
    else:
        model = st.session_state.model
        predictor = st.session_state.predictor
        device = st.session_state.device

# Get image list
if not st.session_state.image_list:
    unlabeled_images, labeled_images = get_unlabeled_images()
    st.session_state.image_list = unlabeled_images
    st.session_state.labeled_images = labeled_images

# Sidebar
with st.sidebar:
    st.header("Progress")
    total_images = len(st.session_state.image_list) + len(
        st.session_state.labeled_images
    )
    labeled_count = len(st.session_state.labeled_images)
    st.metric("Labeled Images", f"{labeled_count}/{total_images}")
    st.progress(labeled_count / total_images if total_images > 0 else 0)

    st.divider()
    st.header("Instructions")
    st.markdown(
        """
    1. **Toggle** between positive (✅) and negative (❌) mode
    2. Click on the image to add points
       - **Green dots**: Positive points (include in mask)
       - **Red dots**: Negative points (exclude from mask)
    3. Click **Generate Mask** to see segmentation
    4. Add more clicks if needed and regenerate
    5. Click **Save & Next** when satisfied
    6. Use **Skip** to skip current image
    7. Use **Clear Clicks** to start over

    **Note:** 
    - Images are scaled (smaller side = 1024px)
    - Segmentation is performed on scaled image
    - Masks are upscaled to 4000x3000 when saved
    """
    )

    st.divider()
    st.info(f"Device: {device.upper()}")
    st.info(f"Working Size: {TARGET_SIZE}px (smaller side)")
    st.info(f"Save Size: {SAVE_SIZE[0]}x{SAVE_SIZE[1]}")

# Check if there are images to label
if len(st.session_state.image_list) == 0:
    st.success("🎉 All images have been labeled!")
    if st.button("Refresh Image List"):
        st.session_state.image_list = []
        st.rerun()
    st.stop()

# Get current image
current_image_path = st.session_state.image_list[st.session_state.current_image_idx]
# st.subheader(f"Image: {current_image_path.name}")
# st.caption(f"Image {st.session_state.current_image_idx + 1} of {len(st.session_state.image_list)}")

# Load and resize image for working
original_image = Image.open(current_image_path).convert("RGB")
original_width, original_height = original_image.size

# Resize image proportionally (smaller side = 1024)
working_image = resize_image_proportional(original_image, TARGET_SIZE)
working_width, working_height = working_image.size

# st.info(
#     f"Original: {original_width}x{original_height} → Working: {working_width}x{working_height} → Save: {SAVE_SIZE[0]}x{SAVE_SIZE[1]}")

# Layout
# Layout
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    st.markdown("### Click on the image to add points")
with col2:
    # Toggle button for click mode
    mode_emoji = "✅" if st.session_state.click_mode == 1 else "❌"
    mode_text = "Positive" if st.session_state.click_mode == 1 else "Negative"
    mode_color = "green" if st.session_state.click_mode == 1 else "red"

    if st.button(f"{mode_emoji} Mode: {mode_text}", use_container_width=True):
        st.session_state.click_mode = (
            1 - st.session_state.click_mode
        )  # Toggle between 0 and 1
        st.rerun()


with col3:
    # Generate mask button
    if st.button(
        "🎯 Generate Mask",
        disabled=len(st.session_state.clicks) == 0,
        use_container_width=True,
    ):
        with st.spinner("Generating segmentation..."):
            # Segment on working image
            mask = segment_object_from_clicks(
                working_image, st.session_state.clicks, predictor, device
            )
            st.session_state.current_mask = mask
        # st.success(f"✓ Mask generated!\nWorking size: {st.session_state.current_mask.shape}")

# Create visualization
viz_image = visualize_segmentation(
    working_image, st.session_state.current_mask, st.session_state.clicks
)

# Use streamlit-image-coordinates for click detection
from streamlit_image_coordinates import streamlit_image_coordinates

# Display clickable image
value = streamlit_image_coordinates(
    viz_image, key=f"image_{current_image_path.stem}_{len(st.session_state.clicks)}"
)

# Handle click
if value is not None:
    x, y = value["x"], value["y"]
    st.session_state.clicks.append(
        (x, y, st.session_state.click_mode)
    )  # Add label to click
    st.rerun()

# Action buttons
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("💾 Save & Next", disabled=st.session_state.current_mask is None):
        # Save the mask (upscaled to SAVE_SIZE)
        mask_path = save_segmentation(
            current_image_path,
            st.session_state.current_mask,
            st.session_state.clicks,
            (working_width, working_height),
            SAVE_SIZE,
        )
        st.success(
            f"✓ Saved to {mask_path.name}\n(Upscaled to {SAVE_SIZE[0]}x{SAVE_SIZE[1]})"
        )

        # Move to next image
        st.session_state.labeled_images.add(current_image_path)
        st.session_state.image_list.pop(st.session_state.current_image_idx)
        if st.session_state.current_image_idx >= len(st.session_state.image_list):
            st.session_state.current_image_idx = 0

        # Reset state
        st.session_state.clicks = []
        st.session_state.current_mask = None
        st.rerun()

with col2:
    if st.button("⏭️ Skip"):
        # Move to next image without saving
        st.session_state.current_image_idx = (
            st.session_state.current_image_idx + 1
        ) % len(st.session_state.image_list)
        st.session_state.clicks = []
        st.session_state.current_mask = None
        st.rerun()

with col3:
    if st.button("🗑️ Clear Clicks"):
        st.session_state.clicks = []
        st.session_state.current_mask = None
        st.rerun()

with col4:
    if st.button("⬅️ Previous"):
        st.session_state.current_image_idx = (
            st.session_state.current_image_idx - 1
        ) % len(st.session_state.image_list)
        st.session_state.clicks = []
        st.session_state.current_mask = None
        st.rerun()
