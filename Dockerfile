FROM ubuntu:18.04

RUN pip install requirement.txt

RUN python init.py
