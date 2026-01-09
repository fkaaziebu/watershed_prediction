# Steps to start the project on ec2 instance (amazon linux 2023):
* ssh into server
* Create install_python_uv.sh and copy the contents from https://github.com/fkaaziebu/watershed_prediction/blob/main/install_python_uv.sh
* Create setup_ec2.py and copy the contents from https://github.com/fkaaziebu/watershed_prediction/blob/main/setup_ec2.py
* Run install_python_uv.sh (eg. ./install_python_uv.sh) on the ec2 terminal
* Create python virtual environment (eg. uv venv)
* Activate the virtual environment (eg. source uv/bin/activate)
* Run setup_ec2.py (eg. python3 setup_ec2.py)

# Steps to setup on linux or unix pc:
* Install python3 and pip
* Create python virtual environment (eg. uv venv)
* Activate the virtual environment (eg. source uv/bin/activate)
* Run setup_ec2.py (eg. python3 setup_ec2.py)
