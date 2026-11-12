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
    tracks = [dict(track_id=n, bbox=[10*n, 10, 10*n+8, 40], world_pos=(7, 4),
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
