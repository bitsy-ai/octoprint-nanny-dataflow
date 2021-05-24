from time import time
from functools import wraps
from apache_beam import DoFn
from apache_beam.metrics import Metrics


def time_distribution(namespace, name):
    return MeasureTimeDecorator(namespace, name)


class MeasureTimeDecorator(object):
    def __call__(self, to_decorate):
        @wraps(to_decorate)
        def decorated(*args, **kwargs):
            start = time()
            result = to_decorate(*args, **kwargs)
            self.time_distribution.update(int((time() - start) * 1000))
            return result

        return decorated

    def __init__(self, namespace, name):
        self.time_distribution = Metrics.distribution(namespace, name)


class FixedWindowMetricStart(DoFn):
    def __init__(self, window_period: int, job_name: str):
        self.window_period = window_period
        self.job_name = job_name

        self.session_count = Metrics.counter(job_name, "session_count")
        self.seconds_elapsed_count = Metrics.counter(job_name, "seconds_elapsed")

    def process(self, element):
        self.session_count.inc()
        self.seconds_elapsed_count.inc(self.window_period)
        yield element


class FixedWindowMetricEnd(DoFn):
    def __init__(self, window_period: int, job_name: str):
        self.window_period = window_period
        self.job_name = job_name

        self.session_count = Metrics.counter(job_name, "session_count")

    def process(self, element):
        self.session_count.dec()
        yield element
