FROM nvcr.io/nvidia/pytorch:21.05-py3

RUN pip install seaborn \
    && pip install PyYAML --upgrade \
    && pip install mlflow --upgrade \
    && pip install hydra-core --upgrade \
    && pip install pytorch-lightning --upgrade \
    && pip install transformers --upgrade \ 
    && pip install autopep8 --upgrade