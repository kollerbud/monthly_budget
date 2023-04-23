From python:3.8-slim

RUN python -m pip install -U pip

WORKDIR /src

ENV PORT=8080

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY ./src ./src

CMD [ "python", "run", 'app.py']