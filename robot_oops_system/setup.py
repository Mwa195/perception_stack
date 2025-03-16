from setuptools import find_packages, setup

package_name = 'robot_oops_system'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mwa',
    maintainer_email='mohamedwael200320@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'streamer = robot_oops_system.streamer:main',
            'semantic_segmentation_dl = robot_oops_system.semantic_segmentation_dl:main',
        ],
    },
)
