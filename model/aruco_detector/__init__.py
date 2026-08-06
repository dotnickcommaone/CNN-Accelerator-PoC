"""MobileNetV2-0.35 based single-class ArUco marker detector."""

from .network import ArucoMobileNetV2, decode_predictions

__all__ = ["ArucoMobileNetV2", "decode_predictions"]
