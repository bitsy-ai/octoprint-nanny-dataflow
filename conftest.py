import pytest
import print_nanny_dataflow

@pytest.fixture
def partial_nested_telemetry_event_kwargs():
    return dict(
        ts=12345,
        client_version = print_nanny_dataflow.__version__,
        event_type="event_type",
        event_data_type="event_data_type",
        session="12345",
        user_id=1,
        device_id=1,
        device_cloudiot_id=1,
        image_height=320,
        image_width=320,
        num_detections=1
    )