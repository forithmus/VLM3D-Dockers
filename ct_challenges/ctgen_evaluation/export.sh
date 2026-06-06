#!/usr/bin/env bash

./build.sh

docker save ctgen-eval | gzip -c > ctgen-eval.tar.gz
