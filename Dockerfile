
FROM python:3.11

WORKDIR /app

COPY requirements.txt /app/

RUN pip install -r requirements.txt

COPY . /app/

EXPOSE 8501


ENV NAME StreamlitApp

CMD ["streamlit", "run", "app.py"]
