import os
from omegaconf import OmegaConf, DictConfig
from glob import glob
import hydra
import subprocess


def make_image_list(data_path):
    image_list = []
    suffix = ['*.jpg', '*.png', '*.JPG', '*.jpeg']
    for suf in suffix:
        image_list += glob(os.path.join(data_path, 'images', suf)) +\
            glob(os.path.join(data_path, 'images_1', suf))

    assert len(image_list) > 0, "No image found"
    image_list.sort()

    f = open(os.path.join(data_path, 'image_list.txt'), 'w')
    for image_path in image_list:
        f.write(image_path + '\n')


@hydra.main(version_base=None, config_path='../confs', config_name='default')
def main(conf: DictConfig) -> None:
    base_dir = os.getcwd()
    print(f'Working directory is {base_dir}')

    data_path = os.path.join(
        base_dir, 'data', conf['dataset_name'], conf['case_name'])
    assert os.path.exists(data_path), data_path
    make_image_list(data_path)

    base_exp_dir = os.path.join(base_dir, 'exp', conf['case_name'])
    os.makedirs(base_exp_dir, exist_ok=True)

    subprocess.run(
        f"git show -s > {base_exp_dir}/git_info.txt", shell=True, check=True)
    subprocess.run(
        f"git diff >> {base_exp_dir}/git_info.txt", shell=True, check=True)

    conf = OmegaConf.to_container(conf, resolve=True)
    conf['dataset']['data_path'] = data_path
    conf['base_dir'] = base_dir
    conf['base_exp_dir'] = base_exp_dir

    save_config_path = os.path.join(base_exp_dir, 'runtime_config.yaml')
    OmegaConf.save(conf, save_config_path)

    binary_path = f"{base_dir}/build/main"
    assert os.path.exists(binary_path) and os.path.isfile(binary_path)
    subprocess.run(f"{binary_path} {save_config_path}", shell=True, check=True)


if __name__ == '__main__':
    main()
