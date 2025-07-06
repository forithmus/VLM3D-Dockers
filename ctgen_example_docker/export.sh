#!/usr/bin/env bash

./build.sh

docker save generatect | gzip -c > GENERATECT.tar.gz
