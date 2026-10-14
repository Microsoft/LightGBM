#!/bin/bash

set -e -u -o pipefail

# install Intel OpenCL support
# https://www.intel.com/content/www/us/en/developer/articles/tool/opencl-drivers.html#philinux
# https://www.intel.com/content/www/us/en/docs/oneapi/installation-guide-linux/2023-0/yum-dnf-zypper.html#GUID-B5018FF2-B9F3-4ADC-9EB6-F99F6BFC7948
# https://www.intel.com/content/www/us/en/developer/articles/technical/intel-cpu-runtime-for-opencl-applications-with-sycl-support.html

cat > /etc/yum.repos.d/oneAPI.repo <<EOF
[oneAPI]
name=Intel® oneAPI repository
baseurl=https://yum.repos.intel.com/oneapi
enabled=1
gpgcheck=1
repo_gpgcheck=1
gpgkey=https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
EOF

yum update -y
yum install -y \
    intel-oneapi-runtime-opencl
