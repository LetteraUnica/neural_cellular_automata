ARG UBUNTU_VERSION=22.04

# This needs to generally match the container host's environment.
ARG ROCM_VERSION=5.6

# Target the CUDA build image
ARG BASE_ROCM_DEV_CONTAINER=rocm/dev-ubuntu-${UBUNTU_VERSION}:${ROCM_VERSION}-complete

FROM ${BASE_ROCM_DEV_CONTAINER} as build

# Unless otherwise specified, we make a fat build.
# List from https://github.com/ggerganov/llama.cpp/pull/1087#issuecomment-1682807878
# This is mostly tied to rocBLAS supported archs.
# If you have a different amd gpu go to the terminal and write rocm-info
# search for the name and write it down there (it should be something like gfx[n])
ARG ROCM_DOCKER_ARCH= gfx1100 

# Set nvcc architecture
ENV GPU_TARGETS=${ROCM_DOCKER_ARCH}
# Enable ROCm
ENV CC=/opt/rocm/llvm/bin/clang
ENV CXX=/opt/rocm/llvm/bin/clang++
ENV LLAMA_HIPBLAS=on

RUN apt-get update && apt-get -y install cmake protobuf-compiler aria2 git python3-dev

RUN pip install ipykernel langchain chromadb 

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6

RUN pip install numpy scipy matplotlib av einops wandb