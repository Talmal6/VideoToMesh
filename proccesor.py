import cv2
import numpy as np
from class_proccesors.object_predictor import ObjectPredictor
from helpers.helper_fuctions import draw_detections, draw_mask_overlay
from helpers.cad_sketch_render import render_mesh_overlay, render_debug_overlay
from mesh_proccesors.mesh_factory import MeshFactory, MeshData
from mesh_proccesors.mesh_tracker import MeshTracker


class Processor:
    def __init__(self, predictor: ObjectPredictor, tracker: MeshTracker = None):
        self.predictor = predictor
        self.tracker = tracker or MeshTracker()
        self.mesh_factory = MeshFactory()

    @staticmethod
    def _shift_mask(mask, dx, dy, frame_shape):
        """
        Shift a mask by (dx, dy) pixels without wrap-around.
        New areas are filled with 0.
        """
        if mask is None:
            return None

        h, w = frame_shape[:2]
        m = np.asarray(mask)
        if m.ndim > 2:
            m = np.squeeze(m)
        if m.ndim != 2:
            return None

        # ensure uint8 binary
        m = (m > 0).astype(np.uint8) * 255

        M = np.array([[1.0, 0.0, float(dx)],
                      [0.0, 1.0, float(dy)]], dtype=np.float32)

        shifted = cv2.warpAffine(
            m, M, (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return shifted

    @staticmethod
    def _ensure_mask_frame_size(mask, frame_shape):
        if mask is None:
            return None
        h, w = frame_shape[:2]
        m = np.asarray(mask)
        if m.ndim > 2:
            m = np.squeeze(m)
        if m.ndim != 2:
            return None
        if m.shape[:2] != (h, w):
            m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        return m

    def proccess_path(self, path):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {path}")
        return cap

    def handle_mesh_tracking(self, prev_frame, frame, prev_meshes, conf_threshold):
        """Track meshes using similarity transform."""
        return self.tracker.track(prev_frame, frame, prev_meshes, conf_threshold)

    def handle_drawing(self, frame, meshes):
        vis = frame.copy()

        # Render all tracked meshes
        if meshes:
            for mesh_data in meshes:
                vis = render_mesh_overlay(vis, mesh_data)
                render_debug_overlay(vis, mesh_data)
            cv2.imshow("video", vis)
            return

        cv2.imshow("video", vis)

    def run(self, path, conf_threshold=0.3, refresh_every=10):
        cap = self.proccess_path(path)

        prev_frame = None
        prev_meshes = []
        frame_idx = 0

        while True:
            finished, frame = cap.read()
            if not finished:
                break

            need_refresh = (
                prev_frame is None
                or not prev_meshes
                or (frame_idx % int(refresh_every) == 0)
            )

            meshes = []

            if need_refresh:
                # Run YOLO detection and build mesh
                dets = self.predictor.predict(frame, conf_threshold)
                if dets:
                    mesh_data = self.mesh_factory.build(dets[0], frame.shape, frame_bgr=frame)
                    if mesh_data is not None:
                        meshes = [mesh_data]
            else:
                # Track existing meshes
                meshes = self.handle_mesh_tracking(prev_frame, frame, prev_meshes, conf_threshold)
                if not meshes:
                    # Fallback to detection if tracking fails
                    dets = self.predictor.predict(frame, conf_threshold)
                    if dets:
                        mesh_data = self.mesh_factory.build(dets[0], frame.shape, frame_bgr=frame)
                        if mesh_data is not None:
                            meshes = [mesh_data]

            if meshes:
                self.handle_drawing(frame, meshes)
            else:
                cv2.imshow("video", frame)

            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break

            prev_frame = frame.copy()
            prev_meshes = meshes
            frame_idx += 1

        cap.release()
        cv2.destroyAllWindows()


def main():
    predictor = ObjectPredictor()
    tracker = MeshTracker()
    p = Processor(predictor, tracker)
    p.run("./data/napolion_vid.mp4", conf_threshold=0.3, refresh_every=10)


if __name__ == "__main__":
    main()
