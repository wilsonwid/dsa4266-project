#!/bin/bash

nvidia-smi
pipenv shell
python3 train.py
deactivate