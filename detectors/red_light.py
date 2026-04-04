import cv2
import numpy as np
from collections import deque


# ── Notebook HSV constants (detect_traffic_light_color) ──────────────────────
_RED_LOWER    = np.array([0,   120,  70])
_RED_UPPER    = np.array([10,  255, 255])
_YELLOW_LOWER = np.array([20,  100, 100])
_YELLOW_UPPER = np.array([30,  255, 255])

_SIGNAL_BGR = {
    'red':    (0,   0,   255),
    'yellow': (0,   255, 255),
    'green':  (0,   255,   0),
}

_SIGNAL_MSG = {
    'red':    "Detected Signal Status: Stop",
    'yellow': "Detected Signal Status: Caution",
    'green':  "Detected Signal Status: Go",
}


class RedLightDetector:
    def __init__(self, stop_line_y: int, traffic_light_roi: tuple,
                 num_frames_avg: int = 10):
        """
        stop_line_y        – fixed fallback Y; also used directly when Hough
                             finds nothing (keeps line stable from frame 1).
        traffic_light_roi  – (x1, y1, x2, y2) of the traffic light crop.
        """
        self.stop_line_y       = stop_line_y
        self.traffic_light_roi = traffic_light_roi
        self.crossed_ids: set  = set()

        # Hough rolling average (notebook: LineDetector deque approach)
        self._y_start_q = deque(maxlen=num_frames_avg)
        self._y_end_q   = deque(maxlen=num_frames_avg)

        # Cached stop-line geometry
        self._avg_slope     = 0.0
        self._avg_intercept = float(stop_line_y)

    # ── Signal colour (notebook: detect_traffic_light_color HSV logic) ────────

    def get_signal_color(self, frame) -> str:
        x1, y1, x2, y2 = self.traffic_light_roi
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return 'green'
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        if cv2.countNonZero(cv2.inRange(hsv, _RED_LOWER,    _RED_UPPER))    > 0:
            return 'red'
        if cv2.countNonZero(cv2.inRange(hsv, _YELLOW_LOWER, _YELLOW_UPPER)) > 0:
            return 'yellow'
        return 'green'

    def is_red(self, frame) -> bool:
        return self.get_signal_color(frame) == 'red'

    # ── Overlay drawing ───────────────────────────────────────────────────────

    def draw_signal_overlay(self, frame, color: str) -> None:
        """
        Draws (in-place):
          • Signal status text – top-left  (matches notebook message style)
          • Traffic-light icon – top-right (3 stacked circles, active one lit)
        """
        bgr = _SIGNAL_BGR[color]
        msg = _SIGNAL_MSG[color]

        # ── Status text (notebook: FONT_HERSHEY_TRIPLEX at top of frame) ──
        cv2.putText(frame, msg,
                    (10, 28), cv2.FONT_HERSHEY_TRIPLEX, 0.75, bgr, 2, cv2.LINE_AA)

        # ── Traffic-light icon: 3 circles in a dark box (top-right) ──────
        w = frame.shape[1]

        # Icon bounding box
        bx1, by1 = w - 70, 8
        bx2, by2 = w - 8, 128
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (20, 20, 20), -1)   # dark fill
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 255, 255), 2)  # white border

        cx = (bx1 + bx2) // 2
        r  = 16
        cy_red    = by1 + 24
        cy_yellow = by1 + 64
        cy_green  = by1 + 104

        # Lit colour vs dim colour for each circle
        col_red    = (0, 0, 255)   if color == 'red'    else (40,  40,  80)
        col_yellow = (0, 255, 255) if color == 'yellow' else (40,  80,  80)
        col_green  = (0, 255, 0)   if color == 'green'  else (0,   60,   0)

        cv2.circle(frame, (cx, cy_red),    r, col_red,    -1)
        cv2.circle(frame, (cx, cy_yellow), r, col_yellow, -1)
        cv2.circle(frame, (cx, cy_green),  r, col_green,  -1)

    # ── Stop-line detection (notebook: LineDetector / Hough approach) ─────────

    def update_stop_line(self, frame) -> None:
        """Update the rolling-average stop-line geometry from Hough lines."""
        width   = frame.shape[1]
        x_start = 0
        x_end   = width - 1

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray  = cv2.GaussianBlur(gray, (7, 7), 0)
        edges = cv2.Canny(gray, 30, 100)
        edges = cv2.erode(cv2.dilate(edges, None, iterations=1), None, iterations=1)

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80,
                                 minLineLength=100, maxLineGap=10)

        if lines is not None:
            for line in lines:
                lx1, ly1, lx2, ly2 = line[0]
                slope     = (ly2 - ly1) / (lx2 - lx1 + np.finfo(float).eps)
                intercept = ly1 - slope * lx1
                y_s       = int(slope * x_start + intercept)
                y_e       = int(slope * x_end   + intercept)
                # Only keep near-horizontal lines close to the stop line
                if abs(slope) < 0.15 and abs(y_s - self.stop_line_y) < 80:
                    self._y_start_q.append(y_s)
                    self._y_end_q.append(y_e)

        if self._y_start_q:
            avg_ys = int(sum(self._y_start_q) / len(self._y_start_q))
            avg_ye = int(sum(self._y_end_q)   / len(self._y_end_q))
            self._avg_slope     = (avg_ye - avg_ys) / (x_end - x_start + np.finfo(float).eps)
            self._avg_intercept = avg_ys - self._avg_slope * x_start
        # else: keep last known geometry (or the initial stop_line_y fallback)

    def _stop_y_at(self, x: int) -> int:
        return int(self._avg_slope * x + self._avg_intercept)

    # ── Violation check ───────────────────────────────────────────────────────

    def check_violation(self, vehicle_box, track_id: int, signal_color: str) -> bool:
        """True (once per track_id) when vehicle bottom crosses stop line on red."""
        if signal_color != 'red' or track_id in self.crossed_ids:
            return False
        coords   = [int(c) for c in vehicle_box]
        x1, x2, y2 = coords[0], coords[2], coords[3]
        if y2 > self._stop_y_at((x1 + x2) // 2):
            self.crossed_ids.add(track_id)
            return True
        return False

    # ── Draw stop line ────────────────────────────────────────────────────────

    def draw_stop_line(self, frame, color: str = 'green') -> np.ndarray:
        """Full-width stop line in the current signal colour."""
        width  = frame.shape[1]
        bgr    = _SIGNAL_BGR.get(color, (0, 255, 0))
        y_l    = self._stop_y_at(0)
        y_r    = self._stop_y_at(width - 1)
        cv2.line(frame, (0, y_l), (width - 1, y_r), bgr, 3)
        cv2.putText(frame, "STOP LINE",
                    (10, y_l - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr, 2)
        return frame
