FROM apache/beam_python3.8_sdk:2.28.0

RUN apt-get update -qq && apt-get -y install \
    ffmpeg

RUN pip install --upgrade pip wheel setuptools

ADD requirements.txt requirements.txt

RUN pip install -r requirements.txt
ADD setup.py setup.py
RUN pip install -e .
ADD print_nanny_dataflow .
