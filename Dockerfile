FROM python:3.8
WORKDIR /app
COPY feature_engg.py feature_engg.py
COPY app.py app.py
COPY assets assets
COPY requirements.txt requirements.txt
COPY templates/home.html templates/home.html
COPY static/style.css static/style.css
RUN pip install --upgrade pip
RUN pip install numpy
RUN pip install pandas
RUN pip install sklearn
RUN pip install pickle-mixin
RUN pip install pyyaml
RUN pip install Flask-gunicorn
RUN pip install -r requirements.txt
ENV PORT 8080
CMD ["gunicorn", "app:app"]
