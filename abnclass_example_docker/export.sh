#!/usr/bin/env bash

./build.sh

docker save ctlipro | gzip -c > CTLIPRO.tar.gz
