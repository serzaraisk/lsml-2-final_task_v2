FROM pytorch/pytorch
ADD . /Website

EXPOSE 5000
ENV FLASK_APP=server.py
RUN pip install -r requirements.txt
ENTRYPOINT [ "python", "Website/server.py" ]
