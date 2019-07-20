#!/usr/bin/env bash

if [[ -d preprocessed ]] && [[ -d raw ]]; then
    echo "data exist"
    exit 0
else
    wget -c https://dataset-bj.cdn.bcebos.com/dureader/dureader_raw.zip
    wget -c https://dataset-bj.cdn.bcebos.com/dureader/dureader_preprocessed.zip
fi

if md5sum --status -c md5sum.txt; then
    unzip dureader_raw.zip
    unzip dureader_preprocessed.zip
else
    echo "download data error!" >> /dev/stderr
    exit 1
fi
