"""Bounded real-inference smoke checks. Run from repository root."""
import ast
import json
import sys
import time
import importlib.metadata as metadata
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def inventory():
    files = [p for folder in ('cv_core', 'scripts', 'evaluation', 'tests')
             for p in Path(folder).rglob('*.py')] + [Path('app_streamlit.py')]
    for p in files:
        ast.parse(p.read_text(encoding='utf-8-sig'))
    videos = {}
    for p in Path('data/videos').glob('*.mp4'):
        c = cv2.VideoCapture(str(p))
        info = {k: c.get(v) for k, v in {
            'fps': cv2.CAP_PROP_FPS, 'width': cv2.CAP_PROP_FRAME_WIDTH,
            'height': cv2.CAP_PROP_FRAME_HEIGHT, 'reported_frames': cv2.CAP_PROP_FRAME_COUNT}.items()}
        n = 0
        while True:
            ok, frame = c.read()
            if not ok:
                break
            assert frame.size > 0
            n += 1
        c.release()
        assert n == info['reported_frames'] and n > 0
        videos[p.name] = dict(info, decoded_frames=n)
    return {'parsed_python_files': len(files), 'videos': videos,
            'versions': {n: metadata.version(n) for n in
                         ['numpy', 'opencv-python', 'scipy', 'torch', 'ultralytics',
                          'streamlit', 'fastapi', 'uvicorn', 'pandas', 'shapely']}}


def web_checks():
    print('web: importing engine', flush=True)
    from fastapi.testclient import TestClient
    from scripts import web_demo as w
    client = TestClient(w.app)
    print('web: dashboard request', flush=True)
    assert client.get('/').status_code == 200
    old = w.engine.use_pose
    print('web: pose route', flush=True)
    r = client.get('/api/toggle_pose')
    assert r.status_code == 200 and w.engine.use_pose == old
    assert client.post('/api/toggle_pose').status_code == 200
    w.engine.toggle_pose(old)
    assert client.post('/api/switch_feed?feed=walking').status_code == 200
    print('web: real frames', flush=True)
    results = []
    for pose in (False, True):
        w.engine.toggle_pose(pose)
        start = time.perf_counter()
        for _ in range(8):
            time.sleep(w.engine.clock.interval + .001)
            jpeg = w.engine.get_frame()
            decoded = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            assert decoded is not None and decoded.shape == (500, 1200, 3)
        results.append({'pose': pose, 'frames': 8,
                        'fps_including_startup': 8 / (time.perf_counter() - start),
                        'tracks': len(w.engine.tracker.tracks)})
        print(f'web: pose={pose} complete', flush=True)
    # Exercise the actual route generator without leaving an infinite HTTP stream open.
    import asyncio
    async def first_chunk():
        response = w.video_stream()
        iterator = response.body_iterator
        chunk = await anext(iterator)
        assert b'Content-Type: image/jpeg' in chunk and b'\xff\xd8' in chunk
        await iterator.aclose()
    asyncio.run(first_chunk())
    w.engine.cap.release()
    return {'dashboard': 'pass', 'local_feed_change': 'pass',
            'GET_mutates_pose': False, 'mjpeg_route_first_chunk': 'pass',
            'real_inference': results}


def pipeline_checks():
    from scripts.run_cv_pipeline import run_cv_engine
    outcomes = {}
    for video in Path('data/videos').glob('*.mp4'):
        output = Path('audit') / (video.stem + '_output.mp4')
        started = time.perf_counter()
        error = None
        try:
            run_cv_engine(str(video), output_video=str(output), show_window=False)
        except cv2.error as exc:
            # Keep the actual cleanup failure in evidence; verify the produced video too.
            error = str(exc)
        elapsed = time.perf_counter() - started
        cap = cv2.VideoCapture(str(output))
        n = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            assert frame.size > 0
            n += 1
        cap.release()
        original = cv2.VideoCapture(str(video))
        expected = int(original.get(cv2.CAP_PROP_FRAME_COUNT))
        original.release()
        assert n == expected and n > 0, (video, n, expected)
        outcomes[video.name] = {'decoded_output_frames': n, 'elapsed_sec': elapsed,
                                'fps': n / elapsed, 'cleanup_error': error}
    return outcomes


def calibrated_checks():
    from ultralytics import YOLO
    from cv_core.geometry.homography import HomographyTransformer
    from cv_core.geometry.zones_3d import MetricZone3D
    from cv_core.tracking.tracker import MultiTargetTracker
    from cv_core.kinetics.pose_classifier import PoseKineticClassifier
    from cv_core.monitoring import ZoneMonitor
    from scripts.run_cv_pipeline import run_cv_engine
    cap = cv2.VideoCapture('data/videos/corridor_tripwire_breach.mp4')
    cap.set(cv2.CAP_PROP_POS_FRAMES, 48)
    model = YOLO('yolov8n-pose.pt')
    tracker, monitor = MultiTargetTracker(), ZoneMonitor()
    h = HomographyTransformer.load_calibration('data/calibration.json')
    # A deliberately broad test zone exercises plumbing, not physical calibration accuracy.
    zone = MetricZone3D('test', 'Test area', 'polygon', [(-1000, -1000), (1000, -1000), (1000, 1000), (-1000, 1000)])
    detections_seen, confirmed, events = 0, 0, []
    clip_path, output_path = Path('audit/pose_input.mp4'), Path('audit/pose_output.mp4')
    writer = cv2.VideoWriter(str(clip_path), cv2.VideoWriter_fourcc(*'mp4v'), 12, (768, 432))
    assert writer.isOpened()
    try:
        for _ in range(24):
            ok, frame = cap.read()
            assert ok
            writer.write(frame)
            result = model(frame, classes=[0], conf=.15, imgsz=480, verbose=False)[0]
            detections = [dict(bbox=box.xyxy[0].cpu().tolist(), confidence=float(box.conf[0]),
                               keypoints=result.keypoints.data[i].cpu().numpy()) for i, box in enumerate(result.boxes)]
            detections_seen += len(detections)
            tracks = tracker.update(detections, h, PoseKineticClassifier(), zones=[zone], elapsed_dt=1/12)
            confirmed += len(tracks)
            _, emitted = monitor.update(tracks, [zone], 1/12, dwell_confirm_sec=.25)
            events.extend(emitted)
    finally:
        writer.release()
        cap.release()
    assert detections_seen > 0 and confirmed > 0, (detections_seen, confirmed)
    assert any(e['event'] == 'ENTER' for e in events)
    assert any(e['event'] == 'LOITER' for e in events)
    run_cv_engine(str(clip_path), calibration_file='data/calibration.json',
                  output_video=str(output_path), show_window=False, use_pose=True)
    output = cv2.VideoCapture(str(output_path))
    n = 0
    while output.read()[0]:
        n += 1
    output.release()
    assert n == 24
    return {'real_person_detections': detections_seen, 'confirmed_track_observations': confirmed,
            'events': events, 'pose_cli_decoded_output_frames': n,
            'limit': 'Legacy calibration explicitly selected; broad test zone does not validate real-world accuracy'}


def streamlit_checks():
    from streamlit.testing.v1 import AppTest
    real_capture = cv2.VideoCapture
    class BoundedCapture:
        def __init__(self, *args, **kwargs):
            self.cap = real_capture(*args, **kwargs)
            self.n = 0
        def isOpened(self):
            return self.cap.isOpened()
        def read(self):
            if self.n >= 8:
                return False, None
            self.n += 1
            return self.cap.read()
        def set(self, *args):
            return False
        def get(self, *args):
            return self.cap.get(*args)
        def release(self):
            return self.cap.release()
    at = AppTest.from_file(str(repo_root / 'app_streamlit.py'), default_timeout=60).run()
    assert not at.exception, list(at.exception)
    with patch.object(cv2, 'VideoCapture', BoundedCapture):
        at.button(key='btn_start_live').click().run(timeout=60)
    assert not at.exception, list(at.exception)
    running = at.session_state['is_running']
    assert not running
    at.button(key='btn_stop_feed').click().run()
    assert not at.exception
    return {'initial_render': 'pass', 'start_real_inference_8_frames': 'pass',
            'disconnect_stops_feed': True, 'stop_button': 'pass'}


if __name__ == '__main__':
    import sys
    import faulthandler
    faulthandler.dump_traceback_later(180)
    result = {}
    funcs = {'inventory': inventory, 'web': web_checks, 'streamlit': streamlit_checks,
             'pipeline': pipeline_checks, 'calibrated': calibrated_checks}
    for name in sys.argv[1:] or funcs:
        try:
            result[name] = funcs[name]()
        except Exception as exc:
            import traceback
            result[name] = {'error': repr(exc), 'traceback': traceback.format_exc()}
        Path(f'audit/runtime_{name}.json').write_text(json.dumps(result[name], indent=2), encoding='utf-8')
        print(json.dumps({name: result[name]}, indent=2), flush=True)
    assert all('error' not in r for r in result.values()), result
    faulthandler.cancel_dump_traceback_later()
