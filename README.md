# Running Mobilenet v2 with Raspberry Pi AI Camera

python3 -m venv --system-site-packages my-first-object-detection
source my-first-object-detection/bin/activate
cd my-first-object-detection
git clone https://github.com/dn-sss/picam-1.git
cd picam-1
pip install --upgrade pip
pip3 install -r ./requirements.txt

python3 ./main.py