# Running Mobilenet v2 with Raspberry Pi AI Camera

python3 -m venv --system-site-packages picam-mobilenet-v2
source picam-mobilenet-v2/bin/activate
git clone https://github.com/dn-sss/picam-1.git
cd picam-1
pip install --upgrade pip
pip3 install -r ./requirements.txt

python3 ./main.py