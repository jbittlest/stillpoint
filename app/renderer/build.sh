#!/bin/sh
# Build the Stillpoint renderer CLI -> app/renderer/.build/sprender
# (Metal shaders are compiled at runtime from shaders/warp.metal: Xcode 26 ships without the offline Metal compiler.)
set -e
cd "$(dirname "$0")"
mkdir -p .build
swiftc -O -swift-version 5 -suppress-warnings main.swift -o .build/sprender
echo "built $(pwd)/.build/sprender"
