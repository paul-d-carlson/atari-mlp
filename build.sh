#!/bin/sh
# build.sh — CC65 build script for ataritron (Atari 8-bit)
# Produces ataritron.xex loadable by Atari DOS or an emulator.

TARGET=atari
OUTPUT=ataritron.xex
SRCS="ataritron.c mlp.c"

cl65 -t ${TARGET} -O --standard c89 -o ${OUTPUT} ${SRCS}

if [ $? -eq 0 ]; then
    echo "Build OK: ${OUTPUT}"
else
    echo "Build FAILED"
    exit 1
fi
