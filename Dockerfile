FROM python:3.12

EXPOSE 8080
WORKDIR /app

COPY . ./

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install google-generativeai
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install av

ENTRYPOINT ["streamlit", "run", "example.py", "--server.port=8080", "--server.address=0.0.0.0"]
