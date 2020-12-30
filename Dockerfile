FROM python:3.8
WORKDIR /app
COPY feature_engg.py feature_engg.py
COPY app.py app.py
COPY assets assets
RUN pip install --upgrade pip
RUN pip install numpy
RUN pip install pandas
RUN pip install sklearn
RUN pip install pickle-mixin
RUN pip install pyyaml
RUN pip install Flask-gunicorn

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app