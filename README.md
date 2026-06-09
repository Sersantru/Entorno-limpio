# PASOS PARA CREAR EL ENTORNO VIRTUAL EN LA JETSON

## Estos pasos son necesarios para este proyecto en concreto si se desea importar ultralytics y easyocr.

Crear entorno virtual con --system-site-packages
	- python3 -m venv .venv --system-site-packages
	- source .venv/bin/activate
Instalar setuptools==59.5.0 con pip
	- pip install setuptools==59.5.0
Instalar torchvision del whl hecho sin dependencias
	- pip install /home/aaeon/Proyectos/torchvision-0.16.1-cp38-cp38-linux_aarch64.whl --no-deps
Instalar opencv-python-headless==4.13.0.92 ignorando el global
	- pip install --ignore-installed opencv-python-headless==4.13.0.92
Instalar "seaborn>=0.11.0" "thop>=0.1.1" "matplotlib>=3.3.0" tal cual para que no lo pida ultralytics (se instalarán solos importlib-resources y contourpy)
	- pip install "seaborn>=0.11.0" "thop>=0.1.1" "matplotlib>=3.3.0"
Instalar ultralytics sin dependencias
pip install ultralytics==8.0.230 --no-deps


Para el OCR:
	- pip install easyocr --no-deps