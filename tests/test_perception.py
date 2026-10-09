import sys
from types import SimpleNamespace

import numpy as np

from sightguard.perception.yolo import convert_result


class FakeTensor:
    def __init__(self, values):
        self.values = np.array(values)

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class FakeBoxes:
    def __init__(self, identities):
        self.xyxy = FakeTensor([[1, 2, 20, 30]])
        self.cls = FakeTensor([0])
        self.conf = FakeTensor([0.8])
        self.id = FakeTensor(identities) if identities is not None else None

    def __len__(self):
        return 1


def test_result_conversion_missing_and_present_ids() -> None:
    for identities, expected in ((None, None), ([12], 12)):
        result = SimpleNamespace(boxes=FakeBoxes(identities), names={0: "person"})
        detections = convert_result(result)
        assert detections[0].track_id == expected
        assert detections[0].class_name == "person"
        assert detections[0].bbox_xyxy == (1, 2, 20, 30)


def test_no_boxes() -> None:
    assert convert_result(SimpleNamespace(boxes=None)) == ()
    assert convert_result(SimpleNamespace(boxes=[])) == ()


def test_yolo_adapter_uses_configuration_and_new_model(monkeypatch, tmp_path) -> None:
    from sightguard.config import PerceptionConfig
    from sightguard.perception.yolo import YoloPerceptor

    created = []
    calls = []

    class FakeModel:
        names = {0: "person", 1: "bicycle", 3: "motorcycle"}

        def __init__(self, path):
            created.append(self)

        def track(self, **kwargs):
            calls.append(kwargs)
            return [SimpleNamespace(boxes=None)]

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeModel))
    model = tmp_path / "fake.pt"
    model.write_bytes(b"fake weights")
    config = PerceptionConfig(model_path=str(model), confidence=0.4, image_size=320)
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    for _ in range(2):
        perceptor = YoloPerceptor(config)
        assert perceptor.process(frame) == ()
    assert len(created) == 2 and created[0] is not created[1]
    assert calls[0]["persist"] is True
    assert calls[0]["tracker"] == "botsort.yaml"
    assert calls[0]["conf"] == 0.4
    assert calls[0]["imgsz"] == 320
    assert calls[0]["device"] == "cpu"
    assert calls[0]["classes"] == [0, 1, 3]
