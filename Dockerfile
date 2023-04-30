From python:3.8-slim

WORKDIR /src
ENV PORT=8080

ENV PYTHONUNBUFFERED True

RUN python -m pip install -U pip

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY ./src ./src

EXPOSE 8080

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 deploy:app