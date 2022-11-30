#!/bin/bash

PROFILE_NAME=bigdata
WORKSPACE_LOCATION=/$(pwd)/app
SCRIPT_FILE_NAME=spark_script.py

PARAMS="-y 1991-1994,1999,2001-2003"

docker run -it -v ~/.aws:/home/glue_user/.aws \
  -v "$WORKSPACE_LOCATION":/home/glue_user/workspace/app/ \
  -e AWS_PROFILE=$PROFILE_NAME -e DISABLE_SSL=true \
  --rm -p 4040:4040 -p 18080:18080 \
  --name glue_spark_submit amazon/aws-glue-libs:glue_libs_3.0.0_image_01 \
  spark-submit /home/glue_user/workspace/app/$SCRIPT_FILE_NAME "$PARAMS"
