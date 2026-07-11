from setuptools import find_packages, setup

package_name = 'quadhrrt_nav'

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
    maintainer='kaushal',
    maintainer_email='kaushalkmrgr8@gmail.com',
    description='QuadHRRT* semantic navigation node integrated with Nav2 FollowPath.',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'quadhrrt_node = quadhrrt_nav.quadhrrt_node:main',
        ],
    },
)
