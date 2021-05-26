FROM gcr.io/kaggle-gpu-images/python@sha256:046f8514e4f3f41fef443911ead054414479ea948e2c0fb074114d43daedd794

RUN pip install mlflow --upgrade \
    && pip install hydra-core --upgrade \
    && pip install pytorch-lightning --upgrade \
    && pip install transformers --upgrade \ 
    && pip install autopep8 --upgrade
