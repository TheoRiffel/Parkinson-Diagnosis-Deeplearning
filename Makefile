# Makefile para criação do ambiente Conda + Poetry + kernel Jupyter

SHELL := /bin/bash
CONDA_SH := $(HOME)/miniconda3/etc/profile.d/conda.sh
ENV_NAME := parkinson
ENV_FILE := environment.yaml

.PHONY: all env poetry kernel clean

all: env poetry kernel
	@echo "(OK) => Tudo pronto! Use 'jupyter notebook' e selecione o kernel 'Python ($(ENV_NAME))'."

## Cria ou atualiza o ambiente Conda a partir do environment.yaml
env:
	@echo "(1/3) Configurando ambiente Conda '$(ENV_NAME)'…"
	@source $(CONDA_SH) && \
	conda config --set channel_priority flexible && \
	if conda env list | grep -qE "^$(ENV_NAME)\s"; then \
	  conda env update -n $(ENV_NAME) -f $(ENV_FILE) --prune; \
	else \
	  conda env create -n $(ENV_NAME) -f $(ENV_FILE); \
	fi

## Configura o Poetry para usar o ambiente atual e instala libs Python
poetry:
	@echo "(2/3) => Instalando dependências Python via Poetry…"
	@source $(CONDA_SH) && \
	conda activate $(ENV_NAME) && \
	poetry config virtualenvs.create false && \
	poetry config virtualenvs.prefer-active-python true && \
	poetry install

## Registra o kernel Jupyter apontando para 'parkinson'
kernel:
	@echo "(3/3) => Registrando kernel Jupyter '$(ENV_NAME)'…"
	@source $(CONDA_SH) && \
	conda activate $(ENV_NAME) && \
	poetry run python -m ipykernel install --user \
	  --name $(ENV_NAME) \
	  --display-name "Python ($(ENV_NAME))"
	
## (Opcional) Remove o kernel e apaga o ambiente
clean:
	@echo "Removendo kernel e ambiente Conda…"
	@source $(CONDA_SH) && \
	jupyter kernelspec uninstall -f $(ENV_NAME) || true && \
	conda env remove -n $(ENV_NAME) || true
