import cv2
import numpy as np


LO1 = np.array([0, 50, 50])
HI1 = np.array([15, 255, 255])

LO2 = np.array([160, 50, 50])
HI2 = np.array([180, 255, 255])

MORPH_KERNEL_SMALL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
MORPH_KERNEL_LARGE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

MIN_AREA = 300
SMOOTH_ALPHA = 0.3


def order_corners(pts):
    pts = pts.reshape(4, 2).astype(float)

    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()

    ordered = np.zeros((4, 2), dtype=float)
    ordered[0] = pts[np.argmin(s)]
    ordered[1] = pts[np.argmin(d)]
    ordered[2] = pts[np.argmax(s)]
    ordered[3] = pts[np.argmax(d)]

    return ordered


def fit_quad(contour):
    peri = cv2.arcLength(contour, True)

    for eps in [0.04, 0.06, 0.08, 0.10, 0.03, 0.02]:
        approx = cv2.approxPolyDP(contour, eps * peri, True)

        if len(approx) == 4:
            return order_corners(approx)

    x, y, w, h = cv2.boundingRect(contour)

    return np.array([
        [x, y],
        [x + w, y],
        [x + w, y + h],
        [x, y + h],
    ], dtype=float)


class Tracker:
    def __init__(self):
        self.lost = 0
        self._prev = None

    def _detect(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask1 = cv2.inRange(hsv, LO1, HI1)
        mask2 = cv2.inRange(hsv, LO2, HI2)
        mask = cv2.bitwise_or(mask1, mask2)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL_SMALL)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_KERNEL_LARGE)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None, None

        contour = max(contours, key=cv2.contourArea)

        if cv2.contourArea(contour) < MIN_AREA:
            return None, None

        corners = fit_quad(contour)
        return corners, contour

    def _smooth(self, corners):
        if self._prev is None:
            return corners

        return SMOOTH_ALPHA * self._prev + (1 - SMOOTH_ALPHA) * corners

    def process(self, frame):
        corners, contour = self._detect(frame)

        if corners is None:
            self.lost += 1
            self._prev = None
            return None, None, None, False

        corners = self._smooth(corners)

        self._prev = corners
        self.lost = 0

        cx = corners[:, 0].mean()
        cy = corners[:, 1].mean()

        x, y, w, h = cv2.boundingRect(contour)

        centroid = (cx, cy)
        bbox = (x, y, w, h)

        return corners, centroid, bbox, True

    @staticmethod
    def draw(frame, corners, centroid, bbox, area_ratio=0.0):
        if bbox is not None:
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

        if corners is not None:
            pts = corners.astype(int)

            labels = ["TL", "TR", "BR", "BL"]
            colors = [(255, 0, 0), (0, 255, 255), (0, 0, 255), (255, 0, 255)]

            for pt, label, color in zip(pts, labels, colors):
                cv2.circle(frame, tuple(pt), 6, color, -1)
                cv2.putText(frame, label, tuple(pt + [4, -4]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

        if centroid is not None:
            cx = int(centroid[0])
            cy = int(centroid[1])
            cv2.circle(frame, (cx, cy), 5, (255, 255, 0), -1)

        cv2.putText(frame, f"area={area_ratio:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        return frame