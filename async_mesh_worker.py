# async_mesh_worker.py
import traceback
import numpy as np

def _drain_queue(q):
    """Keep only the newest item in queue (drop older frames)."""
    last = None
    try:
        while True:
            last = q.get_nowait()
    except Exception:
        pass
    return last

def mesh_worker_loop(
    in_q,
    out_q,
    predictor_ctor,   # callable -> ObjectPredictor
    factory_ctor,     # callable -> MeshFactory
):
    """
    Runs in a separate process:
      - receives (frame_idx, frame_bgr, conf_threshold)
      - runs predictor + factory.build
      - returns (frame_idx, mesh_data or None)
    """
    predictor = predictor_ctor()
    mesh_factory = factory_ctor()

    while True:
        item = in_q.get()
        if item is None:
            break

        # If queue accumulated, keep only the newest
        newer = _drain_queue(in_q)
        if newer is not None:
            item = newer

        try:
            frame_idx, frame_bgr, conf_threshold = item
            dets = predictor.predict(frame_bgr, conf_threshold)
            mesh = None
            if dets:
                mesh = mesh_factory.build(dets[0], frame_bgr.shape, frame_bgr=frame_bgr)

            # Return result (non-blocking behavior: if out_q is full, drop oldest)
            try:
                out_q.put_nowait((frame_idx, mesh))
            except Exception:
                # drop one and try again
                try:
                    _ = out_q.get_nowait()
                except Exception:
                    pass
                try:
                    out_q.put_nowait((frame_idx, mesh))
                except Exception:
                    pass

        except Exception:
            # On error: return None mesh, do not crash worker
            err = traceback.format_exc()
            try:
                out_q.put_nowait((frame_idx if 'frame_idx' in locals() else -1, None, err))
            except Exception:
                pass
