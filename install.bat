set PIP_EXTRA_INDEX_URL=https://pypi.python.org/simple https://download.pytorch.org/whl/cu117

python -m venv venv
./venv/Scripts/activate

pip install -e .