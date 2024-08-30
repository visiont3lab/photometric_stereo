#!/bin/bash

sudo apt install python3-pip python3-venv blender
python3 -m venv ps-venv
source ps-venv/bin/activate
pip3 install -r requirements.txt
