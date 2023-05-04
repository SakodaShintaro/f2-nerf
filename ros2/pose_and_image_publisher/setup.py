from setuptools import setup

package_name = 'pose_and_image_publisher'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='sakoda@keio.jp',
    description='Pose and Image Publisher from f2-nerf data directory',
    license='Apache-2.0 license',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pose_and_image_publisher = pose_and_image_publisher.main:main',
        ],
    },
)
