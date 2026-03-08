FROM python:3.14-slim

WORKDIR /app

COPY azure_claude.py .

ENV AZURE_ENDPOINT=""
ENV AZURE_KEY=""
ENV LISTEN_PORT=9000
ENV LISTEN_HOST=0.0.0.0

EXPOSE 9000

CMD ["python3", "azure_claude.py", "--listen-port", "9000", "--listen-host", "0.0.0.0"]
