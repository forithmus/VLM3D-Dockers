#!/usr/bin/env bash

./build.sh

docker save abnclass | gzip -c > abnclass.tar.gz
