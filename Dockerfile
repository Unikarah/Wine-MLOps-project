FROM python:3.9

EXPOSE 8501

COPY . .

RUN pip3 install -r requirement.txt

ENTRYPOINT ["streamlit", "run", "src/front/WineQuality.py", "--server.port=8501", "--server.address=0.0.0.0"]
