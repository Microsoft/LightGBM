#!/bin/bash

set -e -E -u -o pipefail

# recreates 'configure' from 'configure.ac'
# this script should run on Ubuntu 22.04
AUTOCONF_VERSION=$(cat R-package/AUTOCONF_UBUNTU_VERSION)

echo "Creating 'configure' script with Autoconf ${AUTOCONF_VERSION}"

apt update
apt-get install \
    --no-install-recommends \
    -y \
        autoconf="${AUTOCONF_VERSION}"

cd R-package

autoconf \
    --output configure \
    configure.ac \
    || exit 1

rm -r autom4te.cache || echo "no autoconf cache found"

echo "done creating 'configure' script"
