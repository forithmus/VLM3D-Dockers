#!/usr/bin/env bash

./build.sh

docker save sam2 | gzip -c > totalsegmentator.tar.gz
