import math
import torch


def theta_air(sensor_dim: float, focal_length: float) -> float:
    """
    Compute the full in-air field-of-view (FOV) angle (in radians)
    for one dimension (width or height) under rectilinear projection:
        θ = 2 * arctan((sensor_dim) / (2 * f))
    """
    return 2.0 * math.atan(sensor_dim / (2.0 * focal_length))


def theta_water_approx(theta_air_rad: float, n_water: float = 1.34) -> float:
    """
    Approximate the underwater FOV from the in-air angle using the
    small-angle Snell’s-law first-order correction:
        θ_water ≈ θ_air / n_water
    (Actually the report uses the half-angle form, but dividing the full angle
     also works approximately for small angles.)
    """
    return theta_air_rad / n_water


def scene_dimension(
    D: float, sensor_dim: float, focal_length: float, n_water: float = 1.34
) -> float:
    """
    Compute one scene dimension (width or height) in meters using the
    formulas:
      θ_air = 2 * arctan(sensor_dim / (2 * f))
      θ_water ≈ θ_air / n_water
      scene_dim = 2 * D * tan( θ_water / 2 )
    """
    theta_a = theta_air(sensor_dim, focal_length)
    theta_w = theta_water_approx(theta_a, n_water)
    return 2.0 * D * math.tan(theta_w / 2.0)


def pixel_area_estimate(
    w_px: int,
    h_px: int,
    d: float,
    sensor_w: float,
    sensor_h: float,
    focal_length: float,
    n_water: float = 1.34,
) -> float:
    """
    Estimate the pixel-area conversion factor (PAE), i.e. cm² per pixel, or
    equivalent scaling factor, using your report’s formula:
        PAE = (scene_width × scene_height) / (W_px × H_px)
    Returns: PAE in (m² / pixel). If you want cm², multiply by 10,000.
    """
    sw = scene_dimension(d, sensor_w, focal_length, n_water)
    sh = scene_dimension(d, sensor_h, focal_length, n_water)
    scene_area = sw * sh  # in m²
    # area per pixel (in m²)
    return scene_area / (w_px * h_px)


def object_area_from_mask_tensor(mask: torch.Tensor, pae_m2: float) -> float:
    """
    Given a binary mask as a torch.Tensor (dtype torch.uint8, bool, or {0,1}),
    compute the object’s 2D area in m².

    Args:
      mask: PyTorch tensor of shape (H, W) or possibly (1, H, W) etc.
            Non-zero / True values are considered object pixels.
      pae_m2: Pixel-area estimate (m² per pixel).

    Returns:
      area in m² (float).
    """
    # Flatten or sum across spatial dims
    # Convert mask to boolean/int and sum true pixels
    # If mask dtype is bool, sum gives count; otherwise convert to bool
    if mask.dtype == torch.bool:
        pixel_count = mask.sum().item()
    else:
        # Convert non-zero to 1
        pixel_count = mask.bool().sum().item()
    return pixel_count * pae_m2
