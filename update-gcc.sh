# This file is to update the system gcc to version 4.9
# on the gpu cluster
mkdir $HOME/local/bin
ln -s /opt/rh/devtoolset-3/root/usr/bin/gcc $HOME/local/bin/gcc
ln -s /opt/rh/devtoolset-3/root/usr/bin/g++ $HOME/local/bin/g++
ln -s /opt/rh/devtoolset-3/root/usr/bin/c++ $HOME/local/bin/c++
