"""
Unit conversions for the tennis analysis pipeline.

The tennis court is our metric reference frame — because we know its real-world
dimensions in meters and our keypoint detector locates it in pixels, we can
convert between pixels and meters throughout the pipeline.

Called from downstream modules (analytics, mini_court) whenever we need to
translate pixel measurements into physical units like meters or km/h.
"""

# --- Official ITF tennis court dimensions (in meters) ---
DOUBLES_COURT_LENGTH_M = 23.77   # baseline to baseline
DOUBLES_COURT_WIDTH_M = 10.97    # doubles sideline to doubles sideline
SINGLES_COURT_WIDTH_M = 8.23     # singles sideline to singles sideline
SERVICE_BOX_LENGTH_M = 6.40      # baseline to service line
DOUBLES_ALLEY_WIDTH_M = 1.37     # width of one doubles alley
NET_HEIGHT_CENTER_M = 0.914      # net height at center


def pixels_to_meters(
    pixel_distance: float,
    reference_meters: float,
    reference_pixels: float,
) -> float:
    """Convert a pixel distance to meters using a known reference measurement.

    Example: if the court is 400 px wide and 10.97 m wide in reality, a player
    who moved 40 px moved 40 * (10.97 / 400) = 1.097 m.

    Args:
        pixel_distance: The pixel value to convert.
        reference_meters: A real-world length in meters (e.g. court width).
        reference_pixels: The same length measured in pixels in the video.
    """
    return (pixel_distance / reference_pixels) * reference_meters


def meters_to_pixels(
    meter_distance: float,
    reference_meters: float,
    reference_pixels: float,
) -> float:
    """Convert meters to pixels using the same reference. Inverse of pixels_to_meters.

    Used for: drawing scaled overlays like the mini court where we place players
    at their real-world court position but render at pixel resolution.
    """
    return (meter_distance / reference_meters) * reference_pixels


def mps_to_kmh(mps: float) -> float:
    """Meters per second to kilometers per hour. Multiply by 3.6.

    Derivation: 1 hour = 3600 s, 1 km = 1000 m, so 1 m/s = 3600/1000 = 3.6 km/h.
    """
    return mps * 3.6


def kmh_to_mps(kmh: float) -> float:
    """Kilometers per hour to meters per second. Divide by 3.6."""
    return kmh / 3.6