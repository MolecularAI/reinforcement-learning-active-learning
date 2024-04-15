FROM 378255172879.dkr.ecr.eu-west-1.amazonaws.com/docker.io/continuumio/conda-ci-linux-64-python3.7:01.08.2021

USER root

RUN apt-get update && \
    apt-get -y install rsync procps && \

    tar xf lua-5.1.4.9.tar.bz2 && \
    cd lua-5.1.4.9 && \

    make; make install && \




    tar xf Lmod-8.2.tar.bz2 && \

    make install && \

    apt-get autoremove -y && \
    apt-get clean



    LMOD_VERSION=8.2 \



    USER=reinvent

RUN adduser --disabled-password \
    --uid 1000 \
    --gid 100 \
    --home /home/$USER \
    $USER

COPY . /reinventcli/

WORKDIR /reinventcli

RUN conda env update --name=base --file=reinvent.yml && \
    conda clean -afy && \
    chmod -R "a+rx" /reinventcli

USER reinvent
