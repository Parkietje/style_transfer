#!/bin/bash

docker build -f Dockerfile -t parkietje/style-transfer:production .

docker image push parkietje/style-transfer