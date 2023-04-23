## command
```bash
# Create container
docker compose up -d
docker exec -it f2-nerf-container bash

# Setting
apt install sudo -y
sudo apt install -y zlib1g-dev wget unzip cmake
cd f2-nerf
git submodule update --init --recursive

cd External
wget https://download.pytorch.org/libtorch/cu117/libtorch-cxx11-abi-shared-with-deps-1.13.1%2Bcu117.zip
unzip ./libtorch-cxx11-abi-shared-with-deps-1.13.1+cu117.zip

cd ..
cmake . -B build
cmake --build build --target main --config RelWithDebInfo -j8

sudo apt install -y python3-pip
pip3 install click
pip3 install numpy
pip3 install omegaconf
pip3 install hydra-core

# Run
python scripts/run.py --config-name=wanjinyou dataset_name=example case_name=ngp_fox mode=train
```
