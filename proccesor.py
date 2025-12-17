import cv2
import numpy as np
from pathlib import Path
from class_proccesors.object_predictor import ObjectPredictor
from class_proccesors.object_tracker import ObjectTracker
from mesh_proccesors.shape_classifier import infer_shape_type
from mesh_proccesors.shape_fitter import SymmetricShapeFitter
from mesh_proccesors.mesh_factory import save_obj
from helpers.helper_fuctions import (
    draw_detections,
    draw_mask_overlay,
    draw_shape_fit,
    plot_closeups,
)
from helpers.wireframe_render import (
    generate_mesh_from_fit,
    draw_mesh_wireframe,
    MeshRenderProcessor,
)


class Proccessor:
    def __init__(
        self,
        predictor,
        tracker=None,
        shape_fitter=None,
        render_mesh=False,
        render_every=30,
        mesh_out_dir="outputs/meshes",
        render_out_dir="outputs/renders",
    ):
        self.predictor = predictor
        self.tracker = tracker
        self.shape_fitter = shape_fitter or SymmetricShapeFitter()

        self.render_mesh = bool(render_mesh)
        self.render_every = int(render_every)
        self.mesh_out_dir = Path(mesh_out_dir)
        self.render_out_dir = Path(render_out_dir)
        self.mesh_out_dir.mkdir(parents=True, exist_ok=True)
        if self.render_mesh:
            self.mesh_renderer = MeshRenderProcessor(out_dir=self.render_out_dir)
        else:
            self.mesh_renderer = None

    @staticmethod
    def _shift_mask(mask, dx, dy, shape):
        if mask is None:
            return None
        h, w = shape[:2]
        m = np.roll(mask, (int(dy), int(dx)), axis=(0, 1))
        return cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)

    def _export_and_render_mesh(self, det, frame_idx: int):
        if not self.render_mesh:
            return

        fit = det.get("shape_fit")
        if not fit:
            return

        mesh = generate_mesh_from_fit(fit)
        if mesh is None:
            return

        obj_path = self.mesh_out_dir / f"frame_{frame_idx:06d}.obj"
        save_obj(mesh, obj_path.as_posix())

        # Render pretty view every render_every frames to reduce cost
        if self.mesh_renderer and (frame_idx % self.render_every == 0):
            png_path = self.mesh_renderer.process(obj_path)
            preview = cv2.imread(png_path.as_posix())
            if preview is not None:
                cv2.imshow("mesh_render", preview)

    def _attach_shape_fits(self, frame_shape, detections):
        if not detections or self.shape_fitter is None:
            return

        for det in detections:
            mask = det.get("mask")
            bbox = det.get("bbox")
            if mask is None or bbox is None:
                det["shape_fit"] = None
                continue

            shape_hint = infer_shape_type(det.get("class_name"))
            fit = self.shape_fitter.fit(mask, bbox, shape_hint, frame_shape)
            det["shape_fit"] = fit

    def run(self, path, conf_threshold=0.3, refresh_every=10):
        cap = cv2.VideoCapture(path)

        prev_frame = None
        prev_dets = []
        frame_idx = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            need_refresh = (
                self.tracker is None
                or prev_frame is None
                or not prev_dets
                or frame_idx % refresh_every == 0
            )

            if need_refresh:
                dets = self.predictor.predict(frame, conf_threshold)
            else:
                dets = self.tracker.track(
                    prev_frame, frame, prev_dets, conf_threshold
                )

                if dets and "dx" in dets[0]:
                    dets[0]["mask"] = self._shift_mask(
                        prev_dets[0].get("mask"),
                        dets[0]["dx"],
                        dets[0]["dy"],
                        frame.shape,
                    )

                if not dets:
                    dets = self.predictor.predict(frame, conf_threshold)

            vis = frame.copy()
            if dets:
                self._attach_shape_fits(frame.shape, dets)
                #draw_mask_overlay(vis, dets[0].get("mask"))
                draw_detections(vis, dets)
                for det in dets:
                    draw_shape_fit(
                        vis,
                        det.get("shape_fit"),
                        label_prefix=f"{det.get('class_name', '')}: ",
                    )
                    mesh = generate_mesh_from_fit(det.get("shape_fit"))
                    if mesh is not None:
                        draw_mesh_wireframe(vis, mesh, det.get("shape_fit"))
                    self._export_and_render_mesh(det, frame_idx)
            cv2.imshow("video", vis)
           # plot_closeups(frame, dets)

            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break

            prev_frame = frame
            prev_dets = dets
            frame_idx += 1

        cap.release()
        cv2.destroyAllWindows()


def main():
    predictor = ObjectPredictor()
    tracker = ObjectTracker()
    p = Proccessor(predictor, tracker, render_mesh=True, render_every=30)
    p.run("./data/bottle_vid.mp4")


if __name__ == "__main__":
    main()
