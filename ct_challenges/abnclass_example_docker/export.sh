#!/usr/bin/env bash

./build.sh

docker save abnclass-thin | gzip -c > abnclass-thin.tar.gz
