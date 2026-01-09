"""Video processing pipeline for mesh extraction and AR tracking."""

import os
import time
import queue
import threading
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, List

import cv2
import numpy as np

from detection.predictor import Predictor
from mesh.mesh_manager import MeshManager
from mesh.mesh_proccesors.mesh_handler import Handler
from helpers.renderer import Renderer

logger = logging.getLogger(__name__)


@dataclass
class FramePacket:
    """Container for frame data with timing information."""
    idx: int
    frame: np.ndarray
    t_wall: float
    t_perf: float


@dataclass
class MeshPacket:
    """Container for mesh data with timing information."""
    idx: int
    mesh_object: object
    t_wall: float
    t_perf: float


class LatestValue:
    """Thread-safe storage for the latest value."""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._value = None

    def set(self, value) -> None:
        with self._lock:
            self._value = value

    def get(self):
        with self._lock:
            return self._value


class VidToMesh:
    """
    Multi-threaded video processing pipeline for real-time mesh extraction and AR tracking.
    
    Architecture:
        - Capture thread: Reads frames from video source
        - Mesh thread: Processes frames and generates meshes
        - Render thread (main): Displays results with AR overlay
    """
    
    def __init__(
        self,
        predictor: Predictor,
        handlers: Optional[List[Handler]] = None,
        window_title: str = "Mesh AR Tracker",
        renderer_color: Tuple[int, int, int] = (0, 255, 255),
        renderer_alpha: float = 0.4
    ):
        self.predictor = predictor
        self.mesh_manager = MeshManager(handlers=handlers or [])
        self.renderer = Renderer(
            base_color=renderer_color,
            alpha=renderer_alpha,
            preview_every=1
        )

        self._stop = threading.Event()
        self._frame_q: "queue.Queue[FramePacket]" = queue.Queue(maxsize=2)
        self._latest_frame = LatestValue()
        self._latest_mesh = LatestValue()
        self._last_ui_print = 0.0
        self._win_main = window_title

    def process_path(self, source):
        """Initialize video capture from file path or camera index."""
        if isinstance(source, str):
            if not os.path.exists(source):
                if source.isdigit():
                    source = int(source)
                else:
                    raise FileNotFoundError(f"Video file not found: {source}")
            else:
                cap = cv2.VideoCapture(source)
                if not cap.isOpened():
                    raise RuntimeError(f"Failed to open video file: {source}")
                return cap

        if isinstance(source, int) or source is None:
            source = source if source is not None else 0
            cap = cv2.VideoCapture(source)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                return cap
            raise RuntimeError("No working camera found.")

        raise TypeError(f"Invalid source type: {source}")

    @staticmethod
    def _is_file_source(cap: cv2.VideoCapture) -> bool:
        """Check if the video source is a file (vs. camera)."""
        frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        return frame_count > 0.0

    @staticmethod
    def _get_source_fps(cap: cv2.VideoCapture, fallback: float = 30.0) -> float:
        """Get FPS from video source with validation."""
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 1e-3 or not np.isfinite(fps):
            return fallback
        return max(5.0, min(fps, 240.0))

    @staticmethod
    def _put_drop_old(q: queue.Queue, item) -> None:
        """Put item in queue, dropping oldest if full."""
        try:
            q.put_nowait(item)
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            try:
                q.put_nowait(item)
            except queue.Full:
                pass

    def _capture_loop(self, cap: cv2.VideoCapture, loop: bool = False) -> None:
        """Capture thread: reads frames from video source."""
        is_file = self._is_file_source(cap)
        fps = self._get_source_fps(cap)
        frame_period = 1.0 / max(1.0, fps)

        t0 = time.perf_counter()
        idx = 0

        src_type = "file" if is_file else "camera"
        logger.info(f"[Capture] source={src_type}, fps={fps:.2f}, loop={loop}")

        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                if loop and is_file:
                    logger.info("Looping video...")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    t0 = time.perf_counter() # Reset timer for sync
                    idx = 0
                    continue
                else:
                    self._stop.set()
                    break

            pkt = FramePacket(
                idx=idx,
                frame=frame,
                t_wall=time.time(),
                t_perf=time.perf_counter()
            )
            self._latest_frame.set(pkt)
            self._put_drop_old(self._frame_q, pkt)
            idx += 1

            if is_file:
                target = t0 + idx * frame_period
                sleep_s = target - time.perf_counter()
                if sleep_s > 0:
                    time.sleep(sleep_s)

        cap.release()

    def _mesh_loop(self, conf_threshold: float) -> None:
        """Mesh thread: processes frames and generates meshes."""
        logger.info("[Mesh] worker started")
        
        while not self._stop.is_set():
            try:
                pkt: FramePacket = self._frame_q.get(timeout=0.05)
            except queue.Empty:
                continue

            try:
                det = self.predictor.predict(pkt.frame, conf_threshold=conf_threshold)
                mesh_obj = None
                if det is not None:
                    mesh_obj = self.mesh_manager.get_mesh(det, pkt.frame)

                mp = MeshPacket(
                    idx=pkt.idx,
                    mesh_object=mesh_obj,
                    t_wall=time.time(),
                    t_perf=time.perf_counter()
                )
                self._latest_mesh.set(mp)

            except Exception:
                self._stop.set()
                raise

        logger.info("[Mesh] worker stopped")

    def _render_loop(self) -> None:
        """Render thread (main): displays results with AR overlay."""
        logger.info("Press 'q' or 'ESC' to exit")

        cv2.namedWindow(self._win_main, cv2.WINDOW_NORMAL)
        self.renderer.attach_mouse(self._win_main)

        while not self._stop.is_set():
            frame_pkt: Optional[FramePacket] = self._latest_frame.get()
            if frame_pkt is None:
                time.sleep(0.005)
                continue

            mesh_pkt: Optional[MeshPacket] = self._latest_mesh.get()
            mesh_obj = mesh_pkt.mesh_object if mesh_pkt is not None else None

            vis_frame, _ = self.renderer.render_frame(frame_pkt.frame, mesh_obj)
            cv2.imshow(self._win_main, vis_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                self._stop.set()
                break

            now = time.time()
            if now - self._last_ui_print > 0.5:
                self._last_ui_print = now
                mesh_from = mesh_pkt.idx if mesh_pkt is not None else None
                mesh_ok = "yes" if mesh_obj is not None else "no"
                lag_frames = int(frame_pkt.idx - mesh_from) if mesh_from is not None else None
                logger.info(
                    f"[UI] frame={frame_pkt.idx} mesh_from={mesh_from} "
                    f"lag_frames={lag_frames} mesh={mesh_ok}"
                )

        cv2.destroyAllWindows()

    def run(self, source=0, conf_threshold: float = 0.3, loop: bool = False) -> None:
        """
        Run the video processing pipeline.
        
        Args:
            source: Video file path, camera index, or None for default camera
            conf_threshold: Confidence threshold for object detection
            loop: Whether to loop video source
        """
        try:
            cap = self.process_path(source)
        except Exception as e:
            logger.error(f"Error initializing source: {e}")
            return

        self._stop.clear()

        t_cap = threading.Thread(target=self._capture_loop, args=(cap, loop), daemon=True)
        t_mesh = threading.Thread(
            target=self._mesh_loop,
            args=(conf_threshold,),
            daemon=True
        )

        logger.info(f"Starting processor on source: {source}")

        t_cap.start()
        t_mesh.start()

        try:
            self._render_loop()
        except Exception as e:
            logger.error(f"Error in render loop: {e}")
            raise
        finally:
            self._stop.set()
            t_cap.join(timeout=1.0)
            t_mesh.join(timeout=1.0)
            logger.info("Processing finished")
