FROM quay.io/pypa/manylinux_2_28_x86_64

# ensure that libraries like libc++ built in this image can be found by the linker
#ENV LD_LIBRARY_PATH="/usr/local/lib64:${LD_LIBRARY_PATH}:/usr/local/lib"

# use 'bash' for RUN steps
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

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
   sudo

yum module install --nodocs -y \
   llvm-toolset

yum clean all

rm -rf /var/cache/yum
EOF

RUN <<EOF
# install a newer CMake than what the package manager has
curl -sL https://cmake.org/files/v4.2/cmake-4.2.1-linux-$(arch).sh -o cmake.sh
chmod +x cmake.sh
./cmake.sh --prefix=/usr/local --exclude-subdir
rm -f ./cmake.sh

# build PoCL
git clone \
   --depth 1 \
   --branch v7.1 \
   https://github.com/pocl/pocl.git

# explanations for some flags:
#
#   * -DCMAKE_{C,CXX}_COMPILER: DEVTOOLSET_ROOTPATH is where manylinux puts the gcc toolset
#   * -DLLC_HOST_CPU="x86_64": passed to clang's -march/-mcpu flag. see https://github.com/chromebrew/chromebrew/pull/9176#issuecomment-1891751465
#
cmake \
   -B pocl/build \
   -S pocl \
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
   -DLLC_HOST_CPU="x86-64" \
   -DPOCL_DEBUG_MESSAGES=OFF \
   -DPOCL_INSTALL_ICD_VENDORDIR=/etc/OpenCL/vendors

cmake --build pocl/build -j4

cmake --install pocl/build
cmake --install pocl/build --component icd

# try hand-writing the pocl.icd file in the case where pocl's CMake refused to
if ! test -f /etc/OpenCL/vendors/pocl.icd ; then
   echo "manually setting up pocl.id"
   mkdir -p /etc/OpenCL/vendors
   POCL_LIBRARY=$(
      find /usr/local/lib64 -type f -name 'libOpenCL.so.2.*'
   )
   echo "${POCL_LIBRARY}" > /etc/OpenCL/vendors/pocl.icd
fi
EOF

# Install Java
RUN yum install -y \
      java-1.8.0-openjdk-devel.x86_64 \
 && yum clean all

ENV JAVA_HOME_8_X64=/usr/lib/jvm/java
ENV JAVA_HOME=$JAVA_HOME_8_X64

# Install SWIG
RUN curl -sLk https://sourceforge.net/projects/swig/files/swig/swig-4.0.2/swig-4.0.2.tar.gz/download -o swig.tar.gz \
 && tar -xzf swig.tar.gz \
 && cd swig-4.0.2 \
 && ./configure --prefix=/usr/local --without-pcre \
 && make \
 && make install \
 && cd .. \
 && rm -f ./swig.tar.gz \
 && rm -rf ./swig-4.0.2

# Install miniforge
RUN curl -sL "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-$(uname -m).sh" -o miniforge.sh \
 && chmod +x miniforge.sh \
 && ./miniforge.sh -b -p /opt/miniforge \
 && rm -f ./miniforge.sh \
 && /opt/miniforge/bin/conda clean -a -y \
 && chmod -R 777 /opt/miniforge

ENV CONDA=/opt/miniforge/
