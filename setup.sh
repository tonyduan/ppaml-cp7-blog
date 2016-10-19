#!/bin/bash

# install python and armadillo dependencies

sudo apt-get update
sudo apt-get install git g++ cmake libopenblas-dev liblapack-dev libarpack++2-dev unrar
sudo apt-get install python3 python3-pip

# extract and configure armadillo

wget http://sourceforge.net/projects/arma/files/armadillo-7.400.2.tar.xz
tar xvf armadillo-7.400.2.tar.xz
cd armadillo-7.400.2
cmake .
make
sudo make install
cd ..
rm armadillo-7.400.2.tar.xz

# clone and compile swift

git clone https://github.com/lileicc/swift.git
cd swift
git pull origin quantify-query:quantify-query
git checkout quantify-query
make compile
cd ..
mv swift ..

# install requirements

pip3 install -r requirements.txt

