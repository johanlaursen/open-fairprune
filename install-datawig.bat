:: install visual studio to get a C compiler
python3.8 -m venv venv

set venv=.\venv\Scripts\activate
%venv% && pip install --upgrade pip
%venv% && pip install datawig
%venv% && pip install Cython==0.29.34 numpy==1.17.3 pandas==1.3.5