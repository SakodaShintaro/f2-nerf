#!/bin/bash

set -eux

cd $(dirname $0)/../

wget https://demuc.de/colmap/vocab_tree_flickr100K_words32K.bin
