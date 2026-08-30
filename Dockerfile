FROM python:3.13.15-slim

WORKDIR /data

EXPOSE 8888

COPY requirements.txt /data

RUN pip install --no-cache-dir -r requirements.txt

RUN \
  apt-get update -y && \
  apt-get install -y --no-install-recommends graphviz && \
  apt-get --purge remove -y .\*-doc$ && \
  apt-get clean -y && \
  apt-get autoremove -y && \
  rm -rf /var/lib/apt/lists/*

# RUN groupadd -r docker -g 901 && useradd -u 901 -r -g docker docker
# USER docker
# ENV HOME=/user/docker
# WORKDIR ${HOME}

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "-y"]
