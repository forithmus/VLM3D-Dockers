#!/usr/bin/env bash

./build.sh

docker save reportgen | gzip -c > reportgen.tar.gz
