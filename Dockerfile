FROM ubuntu:18.04

RUN pip install -r requirement.txt

RUN python init.py
