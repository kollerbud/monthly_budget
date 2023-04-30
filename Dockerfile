From python:3.8-slim

WORKDIR /src
ENV PORT=8080
COPY ./src ./src

ENV PYTHONUNBUFFERED True

RUN python -m pip install -U pip

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY ./src ./src

EXPOSE 8080

CMD ["/usr/local/bin/gunicorn", "-w", "4", "-b", "0.0.0.0:8080", "-t", "30", "--pythonpath", "./src", "deploy:app"]