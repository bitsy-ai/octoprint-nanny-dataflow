FROM apache/beam_python3.8_sdk:2.28.0

RUN apt-get update -qq && apt-get -y install \
    ffmpeg

RUN pip install --upgrade pip wheel setuptools
RUN mkdir /app
COPY requirements.txt /app/requirements.txt

RUN pip install -r /app/requirements.txt
COPY setup.py /app/setup.py
COPY print_nanny_dataflow /app/print_nanny_dataflow
# RUN pip install /app
