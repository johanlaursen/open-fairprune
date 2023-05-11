set PIP_EXTRA_INDEX_URL=https://pypi.python.org/simple https://download.pytorch.org/whl/cu117

python3 -m venv venv

set venv=.\venv\Scripts\activate
%venv% && pip install --upgrade pip
%venv% && pip install -e .