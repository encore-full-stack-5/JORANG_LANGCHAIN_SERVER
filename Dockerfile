FROM python:latest

WORKDIR /app/

COPY ./chatbot.py /app/
COPY ./requirements.txt /app/

RUN pip install -r requirements.txt

CMD ["uvicorn", "chatbot:app", "--host", "0.0.0.0", "--port", "80"]