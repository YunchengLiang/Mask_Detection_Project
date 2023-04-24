FROM python:3.8.10
COPY . /app
WORKDIR /app
RUN apt-get update && apt-get install libgl1 -y &&\
    pip install -r requirements.txt
EXPOSE $PORT
CMD gunicorn --workers=5 --bind 0.0.0.0:$PORT app:app --preload -b 0.0.0.0:5000 --timeout 1000