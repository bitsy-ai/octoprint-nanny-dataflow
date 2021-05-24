FROM apache/beam_python3.8_sdk:2.29.0

RUN apt-get update -qq && apt-get -y install \
    ffmpeg

