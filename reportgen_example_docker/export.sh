#!/usr/bin/env bash

./build.sh

docker save ctchat | gzip -c > CTCHAT.tar.gz
