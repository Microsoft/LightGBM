FROM quay.io/pypa/manylinux_2_28_aarch64

# use 'bash' for RUN steps
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

# allow the linker to find OpenCL libraries
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/lib64:/usr/local/lib64:/usr/local/lib"

# install packages
RUN <<EOF
yum update -y

yum install --nodocs -y \
   epel-release

yum install --nodocs -y \
   clang-devel \
   hwloc-devel \
   llvm-devel \
   llvm-static \
   ocl-icd-devel \
   opencl-headers \
   sudo

yum module install -y \
   llvm-toolset

yum clean all

rm -rf /var/cache/yum
EOF

RUN <<EOF
# install a newer CMake than what the package manager has
curl -sL https://cmake.org/files/v4.2/cmake-4.2.1-linux-aarch64.sh -o cmake.sh
chmod +x cmake.sh
./cmake.sh --prefix=/usr/local --exclude-subdir
rm -f ./cmake.sh

# build PoCL
#
# NOTE: If this is updated, check if CL_TARGET_OPENCL_VERSION in cmake/IntegratedOpenCL.cmake
#       needs to be updated (see comments there for links).
git clone \
   --depth 1 \
   --branch v7.1 \
   https://github.com/pocl/pocl.git

# compile support for a bunch of different aarch64 CPUs
#
# Copied from https://github.com/conda-forge/pocl-feedstock/blob/64c2c02d96960160e20cabf1b1392a1752146f29/recipe/build.sh#L58-L62
#
# This support is mostly added for CI...testing device="gpu" codepaths on host CPUs, which tend to be cheaper and more
# available than CI runners with GPUs. But using the host CPU as an OpenCL device can be useful for research as well.
#
AARCH64_CPUS="generic;cortex-a35;cortex-a53;cortex-a55;cortex-a57;cortex-a65;cortex-a72;cortex-a73;cortex-a75;cortex-a76"
AARCH64_CPUS="${AARCH64_CPUS};cyclone;exynos-m3;exynos-m4;exynos-m5;falkor;kryo;neoverse-e1;neoverse-n1;saphira"
AARCH64_CPUS="${AARCH64_CPUS};thunderx;thunderx2t99;thunderxt81;thunderxt83;thunderxt88;tsv110"

# explanations for some flags:
#
#   * -DCMAKE_{C,CXX}_COMPILER: DEVTOOLSET_ROOTPATH is where manylinux puts the gcc toolset
#   * -DENABLE_ICD=ON: select appropriate OpenCL platform at runtime (https://github.com/pocl/pocl/blob/013d2f19f4e8f2e0fd9aedcb70117d6dcc737aa9/doc/sphinx/source/using.rst#installable-client-driver-icd)
#   * -DLLC_HOST_CPU="cortex-a35": passed to clang's -march/-mcpu flag, which is like "support a bunch of Arm CPUs".
#
# TODO: fix /etc/OpenCL/vendors not being populated:
#
#  - needs an ocl-icd>=2.3 (I think?)
#  - almalinux 8 has ocl-icd=2.2.*
#  - almalinux 10 has a 2.3.*, but its GLIBC requirement is too high
#
# relevant CMake output:
#
# -- Checking for module 'ocl-icd>=1.3'
# --   Found ocl-icd, version 2.2.12
# CMake Warning at CMakeLists.txt:82 (message):
#   Requested build with icd, but ICD loader not found! some examples will not
#   work..
# Call Stack (most recent call first):
#   CMakeLists.txt:1518 (setup_cached_var)
# -- Using an ICD loader (override): 0
# -- Run tests with ICD: 0
# -- Using PoCL's OpenCL 3.0 headers
#
# references:
#
#    - https://github.com/pocl/pocl/issues/1280
#
#
cmake \
   -B pocl/build \
   -S pocl \
   -DCLANG_MARCH_FLAG="-mcpu=" \
   -DCMAKE_BUILD_TYPE=release \
   -DCMAKE_C_COMPILER="clang" \
   -DCMAKE_CXX_COMPILER="clang++" \
   -DENABLE_DOXYGEN=OFF \
   -DENABLE_ICD=ON \
   -DENABLE_LLVM=ON \
   -DENABLE_EXAMPLES=OFF \
   -DENABLE_HOST_CPU_DEVICES=ON \
   -DENABLE_HWLOC=ON \
   -DENABLE_POCLCC=ON \
   -DENABLE_SPIRV=ON \
   -DENABLE_TESTS=OFF \
   -DENABLE_VALGRIND=OFF \
   -DINSTALL_OPENCL_HEADERS=OFF \
   -DKERNELLIB_HOST_CPU_VARIANTS="${AARCH64_CPUS}" \
   -DLLC_HOST_CPU=cortex-a35 \
   -DPOCL_DEBUG_MESSAGES=OFF \
   -DPOCL_INSTALL_ICD_VENDORDIR=/etc/OpenCL/vendors

cmake --build pocl/build -j4

cmake --install pocl/build
cmake --install pocl/build --component icd

# try hand-writing the pocl.icd file in the case where pocl's CMake refused to
if ! test -f /etc/OpenCL/vendors/pocl.icd ; then
   mkdir -p /etc/OpenCL/vendors
   POCL_LIBRARY=$(
      find /usr/local/lib64 -type f -name 'libOpenCL.so.2.*'
   )
   echo "${POCL_LIBRARY}" > /etc/OpenCL/vendors/pocl.icd
fi
EOF
