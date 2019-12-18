
# install OpenCV dependencies
sudo apt -y install build-essential cmake git pkg-config libgtk-3-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev \
    libjpeg-dev libpng-dev libtiff-dev gfortran openexr libatlas-base-dev \
    python3-dev python3-numpy python2-dev python-numpy libtbb2 libtbb-dev libdc1394-22-dev \
    python3-pip python-pip python3-venv virtualenv

virtualenv --distribute --no-site-packages venv

source venv/bin/activate
pip install --upgrade pip

pip install -r perception/requirements.txt