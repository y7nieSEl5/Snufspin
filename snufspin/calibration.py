"""Camera geometry and video-bound calibration sidecars."""

from dataclasses import dataclass

import cv2
import numpy as np

from .court import COURT_MODEL
from .tracking import file_sha256
from .video import Video


def _array(value, shape, label):
    array = np.array(value, dtype=np.float64, copy=True)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{label} must be finite with shape {shape}")
    array.flags.writeable = False
    return array


@dataclass(frozen=True)
class Intrinsics:
    K: np.ndarray
    distortion: np.ndarray | None = None

    def __post_init__(self):
        K = _array(self.K, (3, 3), "K")
        if (min(K[0, 0], K[1, 1]) <= 0 or K[0, 1] != 0 or K[1, 0] != 0
                or not np.array_equal(K[2], [0, 0, 1])):
            raise ValueError("K must be a zero-skew OpenCV pinhole matrix with positive focal lengths")
        object.__setattr__(self, "K", K)
        if self.distortion is not None:
            coeffs = np.asarray(self.distortion, dtype=np.float64).reshape(-1)
            if coeffs.size not in (4, 5, 8, 12, 14):
                raise ValueError("expected 4, 5, 8, 12 or 14 OpenCV distortion coefficients")
            object.__setattr__(self, "distortion", _array(coeffs, coeffs.shape, "distortion"))

    def undistort(self, pixels, *, normalized=False):
        pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 1, 2)
        if self.distortion is None and not normalized:
            return pixels.reshape(-1, 2)
        return cv2.undistortPointsIter(
            pixels, self.K, self.distortion, None, None if normalized else self.K,
            (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-12),
        ).reshape(-1, 2)

    def distort(self, pixels):
        pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
        if self.distortion is None:
            return pixels
        normalized = (pixels - self.K[:2, 2]) / np.diag(self.K)[:2]
        return cv2.projectPoints(
            np.column_stack((normalized, np.ones(len(pixels)))), np.zeros(3), np.zeros(3),
            self.K, self.distortion,
        )[0].reshape(-1, 2)

    def to_dict(self):
        return {
            "K": self.K.tolist(),
            "distortion": None if self.distortion is None else self.distortion.tolist(),
        }


@dataclass(frozen=True)
class Camera:
    intrinsics: Intrinsics
    R: np.ndarray
    t: np.ndarray

    def __post_init__(self):
        R = _array(self.R, (3, 3), "R")
        if not np.allclose(R @ R.T, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(R), 1):
            raise ValueError("R must be a proper rotation")
        object.__setattr__(self, "R", R)
        object.__setattr__(self, "t", _array(self.t, (3,), "t"))

    @property
    def center(self):
        return -self.R.T @ self.t

    def depth(self, points):
        return (np.asarray(points) @ self.R[2]) + self.t[2]

    def project(self, points):
        """Project one XYZ point or an array of XYZ points to source pixels."""
        points = np.asarray(points, dtype=np.float64)
        if points.ndim == 0 or points.shape[-1] != 3 or not np.isfinite(points).all():
            raise ValueError("expected finite XYZ points")
        if np.any(self.depth(points) <= 0):
            raise ValueError("cannot project points at zero or negative camera depth")
        return cv2.projectPoints(
            points.reshape(-1, 3), cv2.Rodrigues(self.R)[0], self.t,
            self.intrinsics.K, self.intrinsics.distortion,
        )[0].reshape((*points.shape[:-1], 2))

    def pixel_ray(self, pixel):
        """Return a world-space camera center and unit forward direction."""
        pixel = _array(pixel, (2,), "pixel")
        x, y = self.intrinsics.undistort(pixel, normalized=True)[0]
        direction = self.R.T @ [x, y, 1.0]
        return self.center, direction / np.linalg.norm(direction)


@dataclass(frozen=True)
class CourtMapping:
    """H maps court XY in metres to ideal pixels if intrinsics are supplied."""

    H: np.ndarray
    intrinsics: Intrinsics | None = None

    def __post_init__(self):
        H = _array(self.H, (3, 3), "H")
        if np.linalg.matrix_rank(H) != 3:
            raise ValueError("court homography must be invertible")
        object.__setattr__(self, "H", H)

    @classmethod
    def from_camera(cls, camera):
        return cls(camera.intrinsics.K @ np.column_stack((camera.R[:, :2], camera.t)), camera.intrinsics)

    def project(self, xy):
        pixels = _homography(self.H, xy)
        return pixels if self.intrinsics is None else self.intrinsics.distort(pixels).reshape(pixels.shape)

    def pixel_to_court(self, pixel):
        """Map a surface pixel, such as a visible bounce, to world XYZ."""
        pixel = _array(pixel, (2,), "pixel")
        if self.intrinsics is not None:
            pixel = self.intrinsics.undistort(pixel)[0]
        return np.append(_homography(np.linalg.inv(self.H), pixel), 0.0)


def _homography(H, xy):
    xy = np.asarray(xy, dtype=np.float64)
    coords = xy @ H[:, :2].T + H[:, 2]
    if np.any(np.isclose(coords[..., 2], 0, atol=1e-12 * np.linalg.norm(H), rtol=0)):
        raise ValueError("point at the homography horizon")
    return coords[..., :2] / coords[..., 2:]


@dataclass(frozen=True)
class Calibration:
    mapping: CourtMapping
    source: dict
    camera: Camera | None = None

    def to_report(self, video: Video) -> dict:
        return {
            "schema_version": 1,
            "court_model": COURT_MODEL,
            "video_sha256": file_sha256(video.path),
            "width": video.width,
            "height": video.height,
            "frame_count": len(video.timestamps_s),
            "source": self.source,
            "court_mapping": {
                "H": self.mapping.H.tolist(),
                "intrinsics": None if self.mapping.intrinsics is None else self.mapping.intrinsics.to_dict(),
            },
            "camera": None if self.camera is None else {
                **self.camera.intrinsics.to_dict(),
                "R": self.camera.R.tolist(),
                "t": self.camera.t.tolist(),
            },
        }


def read_calibration(video: Video, report: dict) -> Calibration:
    if report["schema_version"] != 1 or report["court_model"] != COURT_MODEL:
        raise ValueError("unsupported calibration schema or court model")
    if ((report["width"], report["height"], report["frame_count"])
            != (video.width, video.height, len(video.timestamps_s))
            or report["video_sha256"] != file_sha256(video.path)):
        raise ValueError("calibration was generated for a different video")
    mapping = report["court_mapping"]
    intrinsics = mapping["intrinsics"]
    camera = report["camera"]
    return Calibration(
        CourtMapping(mapping["H"], None if intrinsics is None else Intrinsics(**intrinsics)),
        report["source"],
        None if camera is None else Camera(
            Intrinsics(camera["K"], camera["distortion"]), camera["R"], camera["t"],
        ),
    )
