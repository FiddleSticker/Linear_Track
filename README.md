# Linear_Track

1) install Python version(3.8.10)

2) clone Cobel & select the aj_drqn_agent
```
git clone https://gitlab.ruhr-uni-bochum.de/cns/1-frameworks/CoBeL-RL.git
git checkout aj_gqrn_agent
```

3) clone Linear_track
```
git clone https://github.com/FiddleSticker/Linear_Track.git
```
4) create venv in Linear_track, activate venv & upgrade pip
```
python3 -m venv venv
source ./venv/bin/activate
pip install -U pip
```
5) install cobel
```
pip install path/to/CoBeL-RL[keras-rl,tensorflow,torch]
```
6) install requirements.txt
```
pip install -r requirements.txt
```
