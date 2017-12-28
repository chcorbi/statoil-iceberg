#!/bin/bash

echo "User: ${USER}"
sudo docker rm ${USER}_iceberg

echo "Choose a port for Tensorboard: "
read tport

echo "Choose a port for Jupyter Notebook: "
read jport

sudo docker run -it --runtime=nvidia --name ${USER}_iceberg -p ${tport}:6006 -p ${jport}:8888 -v ${HOME}:/workspace semseg bash
