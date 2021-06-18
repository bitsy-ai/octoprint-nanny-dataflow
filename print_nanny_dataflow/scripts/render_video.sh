#!/bin/bash

echo "Processing $@"
while getopts ":i:o:s:c:" opt; do
  case ${opt} in
    i )
      INPUT_PATH="$OPTARG"
      ;;
    s )
      SESSION="$OPTARG"
      ;;
    o )
      OUTPUT_FILE="$OPTARG"
      ;;
    c )
      COPY_OUTPUT_FILE="$OPTARG"
      ;;
    \? ) echo "Usage: cmd [-i] [-s] [-o] [-c] [-b]"
      ;;
  esac
done
shift $((OPTIND -1))

TMP_DIR=$(mktemp -d -t render-video-XXXXXXXXXX)

echo "Copying $INPUT_PATH to $TMP_DIR"
gsutil -m cp -r "$INPUT_PATH" "$TMP_DIR"
ffmpeg -pattern_type glob -i "$TMP_DIR/$SESSION/*.jpg" "$TMP_DIR/annotated_video.mp4"
echo "Copying $TMP_DIR/annotated_video.mp4 to $OUTPUT_FILE"
gsutil -m cp "$TMP_DIR/annotated_video.mp4" "$OUTPUT_FILE"
echo "Copying $OUTPUT_FILE to $COPY_OUTPUT_FILE"
gsutil -m cp "$OUTPUT_FILE" "$COPY_OUTPUT_FILE"
trap '{ rm -rf -- "$TMP_DIR"; }' EXIT
