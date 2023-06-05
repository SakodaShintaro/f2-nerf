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
pip3 install click numpy omegaconf hydra-core

# Prepare directory
ln -s /root/data/learn_result/ ./exp

# Run
python3 scripts/run.py --config-name=wanjinyou dataset_name=example case_name=ngp_fox mode=train
```

## install ros2
see https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html

```bash
sudo apt install software-properties-common -y
sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update
sudo apt upgrade -y
sudo apt install -y ros-humble-ros-base
sudo apt install -y ros-humble-cv-bridge
sudo apt install -y ros-humble-geographic-msgs
sudo apt install -y ros-humble-rmw-cyclonedds-cpp
sudo apt install -y ros-humble-demo-nodes-cpp
sudo apt install -y ros-humble-rosbag2-storage-mcap
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
echo "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp" >> ~/.bashrc
```

## Others
```bash
sudo apt install -y ffmpeg colmap python3-colcon-common-extensions ccache vim gdb

sudo add-apt-repository ppa:borglab/gtsam-release-4.1
sudo apt update
sudo apt install -y libgtsam-dev libgtsam-unstable-dev

# add .bashrc
export CC="/usr/lib/ccache/gcc"
export CXX="/usr/lib/ccache/g++"
```
