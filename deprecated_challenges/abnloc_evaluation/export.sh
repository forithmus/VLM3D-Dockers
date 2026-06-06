#!/usr/bin/env bash

./build.sh

docker save abnloc | gzip -c > abnloc.tar.gz
