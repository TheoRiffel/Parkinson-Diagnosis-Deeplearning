.PHONY: setup install kernel notebook clean

all: setup install kernel

# 1) Configura Poetry para criar um .venv local e usa o Python que você apontou
setup:
	poetry config virtualenvs.create true
	poetry config virtualenvs.in-project true
	poetry config virtualenvs.path --unset
	poetry config virtualenvs.use-poetry-python false
	poetry env remove --all || true
	poetry env use $(shell which python3)

# 2) Instala tudo no .venv local
install:
	poetry install
	poetry add ipykernel
	poetry run python -m ipykernel install \
		--user \
		--name parkinson \
		--display-name "Python (parkinson)"

# 3) Registra o kernel Jupyter com nome “parkinson”
kernel:
	. ./.venv/bin/activate && \
	python -m ipykernel install \
	   --user \
	   --name=parkinson \
	   --display-name="Python (parkinson)"

# 4) Ativa e abre o Jupyter
notebook:
	. ./.venv/bin/activate && \
	poetry run jupyter notebook

# 5) Limpa o .venv e o kernel (opcional)
clean:
	poetry run jupyter kernelspec uninstall -f parkinson || true
	rm -rf ./.venv
