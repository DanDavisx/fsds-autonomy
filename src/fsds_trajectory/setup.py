from setuptools import find_packages, setup

package_name = 'fsds_trajectory'

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
    maintainer='dan',
    maintainer_email='dan@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
        'trajectory_publisher = fsds_trajectory.trajectory_publisher:main',
        'tf_from_odom = fsds_trajectory.tf_from_odom:main',
        ],
    },
)
