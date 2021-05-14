FROM apache/beam_python3.8_sdk:2.28.0

RUN apt-get update -qq && apt-get -y install \
    ffmpeg

ADD models .
ADD scripts .