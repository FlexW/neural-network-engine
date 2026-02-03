#!/bin/sh

set -e

out_dir=".out"

cc="clang"
protoc="protoc"

# Ensure clang is installed
if ! command -v $cc >/dev/null 2>&1; then
    echo "Error: clang is not installed. Please install clang compiler." >&2
    exit 1
fi

# Ensure protoc is installed
if ! command -v $protoc >/dev/null 2>&1; then
    echo "Error: protoc is not installed. Please install Protocol Buffers compiler." >&2
    exit 1
fi

cflags_common="\
    -std=c11 \
    -g \
    -fno-omit-frame-pointer \
    -pedantic \
    -pedantic-errors \
    -Wall \
    -Wextra \
    -Werror \
    -Wshadow \
    -Wconversion \
    -Wdouble-promotion \
    -Wmissing-prototypes \
    -Wstrict-prototypes \
    -Wno-gnu-statement-expression-from-macro-expansion \
    -fPIC \
    -Isrc \
    -I$out_dir \
    -I$out_dir/src \
    -DHAVE_GFX=0"

ldflags_common="-lm -ldl -lprotobuf-c"

if [ "$1" = "release" ]; then
    cflags_common="$cflags_common -O3 -ffast-math -DNDEBUG"
else
    cflags_common="$cflags_common -O0 -DDEBUG"
fi

mkdir -p $out_dir

$protoc --c_out=$out_dir src/onnx-ml.proto

$cc $cflags_common -c src/base.c -o $out_dir/base.o
$cc $cflags_common -c src/main.c -o $out_dir/main.o
$cc $cflags_common -c $out_dir/src/onnx-ml.pb-c.c -o $out_dir/onnx-ml.pb-c.o

$cc $ldflags_common \
    $out_dir/base.o \
    $out_dir/main.o \
    $out_dir/onnx-ml.pb-c.o \
    -o $out_dir/neural_net_engine
