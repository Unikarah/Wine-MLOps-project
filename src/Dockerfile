FROM python:3.9

EXPOSE 5002

COPY . .

RUN pip3 install -r requirement.txt

ENTRYPOINT ["flask", "run", "--host=0.0.0.0", "--port", "5002"]
