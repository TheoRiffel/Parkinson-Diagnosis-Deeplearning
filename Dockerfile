# Dockerfile para o projeto
# Base: imagem Jupyter com Conda e pacotes científicos
FROM jupyter/scipy-notebook:python-3.11

# 1. Troca para root para configurações administrativas
USER root

# 2. Configura prioridade de canais Conda para instalação flexível
RUN conda config --set channel_priority flexible

# 3. Copia o arquivo de especificação de ambiente Conda para dentro da imagem
COPY environment.yaml /tmp/environment.yaml

# 4. Cria o ambiente Conda definido em environment.yaml e limpa cache
RUN conda env create -f /tmp/environment.yaml \
    && conda clean -afy

# 5. Registra o novo ambiente Conda como kernel do Jupyter
RUN /opt/conda/envs/t1_rnap_parkinson/bin/python \
      -m ipykernel install --sys-prefix \
      --name t1_rnap_parkinson \
      --display-name "Python (t1_rnap_parkinson)"

# 6. Ajusta PATH para usar o Python do novo ambiente por padrão
ENV PATH /opt/conda/envs/t1_rnap_parkinson/bin:$PATH

# 7. Retorna ao usuário não-root padrão (jovyan)
USER jovyan

# 8. Define diretório de trabalho onde estão seus notebooks e código
WORKDIR /home/jovyan/work

# 9. Expõe a porta onde o Jupyter estará disponível
EXPOSE 8888

# 10. Comando padrão para iniciar o servidor Jupyter sem token de acesso
CMD ["start-notebook.sh", "--NotebookApp.token=''", "--NotebookApp.allow_origin='*'"]
