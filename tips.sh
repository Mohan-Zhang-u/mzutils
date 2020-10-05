#!/usr/bin/env bash

rm -rf dist
rm -rf build
rm -rf mzutils.egg-info
python setup.py bdist_wheel
python -m twine upload dist/*
pip install --upgrade mzutils