from io import BytesIO
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np


def test_web_access_controls_and_read_only_pose_status(monkeypatch):
    from fastapi.testclient import TestClient
    from scripts import web_demo as web
    monkeypatch.delenv('TRIPWIRE_TOKEN', raising=False)
    remote = TestClient(web.app, client=('203.0.113.7', 1234))
    assert remote.get('/').status_code == 403
    assert remote.get('/stream').status_code == 403
    monkeypatch.setenv('TRIPWIRE_TOKEN', 'test-only-password')
    assert remote.get('/').status_code == 401
    assert remote.get('/', auth=('operator', 'wrong')).status_code == 401
    assert remote.get('/', auth=('operator', 'test-only-password')).status_code == 200
    local = TestClient(web.app)
    local.auth = ('operator', 'test-only-password')
    original = web.engine.use_pose
    try:
        assert local.get('/api/toggle_pose').json()['use_pose'] == original
        assert web.engine.use_pose == original
        assert local.post('/api/switch_feed?feed=unknown').status_code == 400
        assert local.post('/api/toggle_pose').status_code == 200
    finally:
        web.engine.toggle_pose(original)


def test_viewers_share_inference_and_unavailable_pose_stays_off():
    from scripts.web_demo import WebStreamEngine
    engine = WebStreamEngine()
    try:
        engine.detector_standard = Mock(return_value=[])
        engine.detector_pose = None
        assert engine.toggle_pose(True) is False
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2) as workers:
            first, second = list(workers.map(lambda _: engine.get_frame(), range(2)))
        assert first == second
        assert engine.detector_standard.call_count == 1
    finally:
        engine.cap.release()


APP_SCRIPT = str(Path(__file__).resolve().parent.parent / 'app_streamlit.py')


def test_upload_same_name_new_content_and_rerun_cleanup():
    from streamlit.testing.v1 import AppTest
    from streamlit.delta_generator import DeltaGenerator
    current = BytesIO(b'first video')
    current.name = 'clip.mp4'
    def upload(label, **kwargs):
        return current if label == 'Upload video file' else None
    at = AppTest.from_file(APP_SCRIPT, default_timeout=30).run()
    with patch.object(DeltaGenerator, 'file_uploader', side_effect=upload):
        at.selectbox[1].select('Upload Custom MP4').run()
        assert not at.exception
        first_path = Path(at.session_state['_upload_path'])
        assert first_path.read_bytes() == b'first video'
        at.run()
        assert Path(at.session_state['_upload_path']) == first_path
        current = BytesIO(b'second video')
        current.name = 'clip.mp4'
        at.run()
        second_path = Path(at.session_state['_upload_path'])
        assert second_path != first_path and not first_path.exists()
        assert second_path.read_bytes() == b'second video'
    at.session_state['_upload_dir'].cleanup()


def test_streamlit_records_two_people_once_and_exports_after_stop():
    from streamlit.testing.v1 import AppTest
    from streamlit.delta_generator import DeltaGenerator
    from cv_core.tracking.tracker import MultiTargetTracker
    calibration = BytesIO(Path('data/calibration.json').read_bytes())
    class Capture:
        def __init__(self, *args):
            self.n = 0
            self.open = True
        def isOpened(self):
            return self.open
        def get(self, key):
            return {cv2.CAP_PROP_FPS: 2, cv2.CAP_PROP_FRAME_WIDTH: 640,
                    cv2.CAP_PROP_FRAME_HEIGHT: 480, cv2.CAP_PROP_POS_MSEC: self.n*500}.get(key, 0)
        def read(self):
            self.n += 1
            return (True, np.zeros((480, 640, 3), np.uint8)) if self.n <= 8 else (False, None)
        def set(self, *args):
            return False
        def release(self):
            self.open = False
    tracks = [dict(track_id=n, bbox=[10*n, 10, 10*n+8, 40], world_pos=(6.5, 2.8),
                   speed=0, posture='standing', threat_level=1) for n in (1, 2)]
    raw_frames = []
    def detector(frame, **kwargs):
        raw_frames.append(frame.copy())
        return []
    at = AppTest.from_file(APP_SCRIPT, default_timeout=30).run()
    with patch.object(DeltaGenerator, 'file_uploader', return_value=calibration), \
         patch.object(cv2, 'VideoCapture', Capture), \
         patch.object(MultiTargetTracker, 'update', return_value=tracks), \
         patch('ultralytics.YOLO', return_value=detector):
        at.button(key='btn_start_live').click().run(timeout=30)
    assert not at.exception, list(at.exception)
    events = at.session_state['incident_log']
    assert sum(e['Event'] == 'ENTER' for e in events) == 2
    assert sum(e['Event'] == 'LOITER' for e in events) == 2
    assert {s['track_id'] for s in at.session_state['evidence_locker']} == {1, 2}
    assert len(at.session_state['evidence_locker']) == 2
    assert raw_frames and all(not frame.any() for frame in raw_frames)
    at.button(key='btn_stop_feed').click().run()
    assert not at.exception
    assert len(at.get('download_button')) >= 2


def test_desktop_disconnected_quit_releases_capture():
    from scripts.demo_gui import TripwireInteractiveDemo
    demo = TripwireInteractiveDemo.__new__(TripwireInteractiveDemo)
    demo.cap = Mock()
    demo.cap.read.return_value = (False, None)
    demo.paused = False
    demo.current_video_key = '0'
    demo.videos = {'0': '0', '1': 'unused.mp4'}
    demo.conf_threshold = .35
    demo.speed_alarm_kmh = 4
    with patch.object(cv2, 'namedWindow'), patch.object(cv2, 'resizeWindow'), \
         patch.object(cv2, 'createTrackbar'), patch.object(cv2, 'imshow'), \
         patch.object(cv2, 'waitKey', return_value=ord('q')), patch.object(cv2, 'destroyAllWindows'):
        demo.run()
    demo.cap.release.assert_called_once()


def test_streamlit_corridor_tripwire_breach_incident_pipeline():
    from streamlit.testing.v1 import AppTest
    from streamlit.delta_generator import DeltaGenerator
    from cv_core.tracking.tracker import MultiTargetTracker
    calibration = BytesIO(Path('data/calibration.json').read_bytes())

    class CrossingCapture:
        def __init__(self, *args):
            self.n = 0
            self.open = True
        def isOpened(self):
            return self.open
        def get(self, key):
            return {cv2.CAP_PROP_FPS: 12, cv2.CAP_PROP_FRAME_WIDTH: 768,
                    cv2.CAP_PROP_FRAME_HEIGHT: 432, cv2.CAP_PROP_POS_MSEC: self.n * 83.3}.get(key, 0)
        def read(self):
            self.n += 1
            # 6 frames: target crosses from Y=2.5 to Y=2.1 (across tripwire at Y=2.3)
            return (True, np.zeros((432, 768, 3), np.uint8)) if self.n <= 6 else (False, None)
        def set(self, *args):
            return False
        def release(self):
            self.open = False

    # Simulate target walking across Y=2.3 (frame 1 at Y=2.6, frame 2 at Y=2.2)
    track_seq = [
        [dict(track_id=1, bbox=[100, 100, 150, 300], world_pos=(6.0, 2.6), speed=3.0, posture='walking', threat_level=1)],
        [dict(track_id=1, bbox=[100, 100, 150, 300], world_pos=(6.0, 2.2), speed=3.0, posture='walking', threat_level=1)],
        [dict(track_id=1, bbox=[100, 100, 150, 300], world_pos=(6.0, 2.1), speed=3.0, posture='walking', threat_level=1)],
        [dict(track_id=1, bbox=[100, 100, 150, 300], world_pos=(6.0, 2.0), speed=3.0, posture='walking', threat_level=1)],
        [dict(track_id=1, bbox=[100, 100, 150, 300], world_pos=(6.0, 1.9), speed=3.0, posture='walking', threat_level=1)],
        [dict(track_id=1, bbox=[100, 100, 150, 300], world_pos=(6.0, 1.8), speed=3.0, posture='walking', threat_level=1)],
    ]
    seq_iter = iter(track_seq)

    def mock_tracker_update(*args, **kwargs):
        return next(seq_iter, [])

    at = AppTest.from_file(APP_SCRIPT, default_timeout=30).run()
    with patch.object(DeltaGenerator, 'file_uploader', return_value=calibration), \
         patch.object(cv2, 'VideoCapture', CrossingCapture), \
         patch.object(MultiTargetTracker, 'update', side_effect=mock_tracker_update), \
         patch('ultralytics.YOLO', return_value=lambda frame, **kw: []):
        at.button(key='btn_start_live').click().run(timeout=30)

    assert not at.exception, list(at.exception)
    events = at.session_state['incident_log']
    tripwire_events = [e for e in events if e.get('Zone') == 'tw_chk' and e.get('Event') in ('CROSS_IN', 'CROSS_OUT')]
    assert len(tripwire_events) >= 1, f"Expected tripwire event, got: {events}"
    assert any('CHECKPOINT TRIPWIRE' in e.get('Incident Reason', '') for e in tripwire_events)

    locker = at.session_state['evidence_locker']
    assert len(locker) >= 1, "Evidence locker snapshot expected on breach"
    assert any(s.get('track_id') == 1 for s in locker)
    assert any('CHECKPOINT TRIPWIRE' in s.get('reason', '') for s in locker)


def test_streamlit_browser_camera_flow():
    """Verify in-browser client camera input path runs without exceptions."""
    from streamlit.testing.v1 import AppTest
    from streamlit.delta_generator import DeltaGenerator

    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    _, enc = cv2.imencode('.jpg', dummy_img)
    camera_buffer = BytesIO(enc.tobytes())

    at = AppTest.from_file(APP_SCRIPT, default_timeout=30).run()
    with patch.object(DeltaGenerator, 'camera_input', return_value=camera_buffer), \
         patch('ultralytics.YOLO', return_value=lambda frame, **kw: []):
        at.selectbox[1].select('Browser Web Camera (Any Laptop / Cloud / Remote)').run()

    assert not at.exception, list(at.exception)

