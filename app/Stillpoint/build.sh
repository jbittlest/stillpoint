#!/bin/sh
# Build Stillpoint.app -> app/Stillpoint/build/Stillpoint.app (ad-hoc signed, macOS 14+, arm64).
#   ./build.sh                  release build
#   ./build.sh --debug          -Onone, faster compile
#   BUILD_DIR=/some/dir ./build.sh   build the bundle somewhere else (e.g. a scratch copy for headless tests)
# The app compiles shaders/warp.metal at runtime and drives the Python engine through engine/stillpoint/app_bridge.py;
# export needs app/renderer/.build/sprender (app/renderer/build.sh).
set -e
cd "$(dirname "$0")"
OPT="-O"
[ "$1" = "--debug" ] && OPT="-Onone"
APP="${BUILD_DIR:-build}/Stillpoint.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
swiftc $OPT -swift-version 5 -parse-as-library -target arm64-apple-macos14.0 \
    -framework SwiftUI -framework AVFoundation -framework Metal -framework CoreImage -framework AppKit \
    Sources/*.swift -o "$APP/Contents/MacOS/Stillpoint"
cp Resources/Info.plist "$APP/Contents/Info.plist"
# Remember where the engine lives (EngineConfig.defaultPath), so builds made elsewhere (BUILD_DIR) still find it.
/usr/libexec/PlistBuddy -c "Add :StillpointEngineRoot string $(cd ../.. && pwd)" "$APP/Contents/Info.plist"
if [ -f Resources/AppIcon.icns ]; then cp Resources/AppIcon.icns "$APP/Contents/Resources/AppIcon.icns"; fi
printf 'APPL????' > "$APP/Contents/PkgInfo"
xattr -cr "$APP"
codesign --force --sign - --timestamp=none "$APP" 2>/dev/null
codesign --verify "$APP"
echo "built $(cd "$(dirname "$APP")" && pwd)/Stillpoint.app"
