From python:3.8-slim

WORKDIR /src
ENV PORT=8080

RUN python -m pip install -U pip

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY ./src ./src
COPY ./saved_models ./saved_models

EXPOSE 8080 8080

CMD ["python", "src/main.py"]