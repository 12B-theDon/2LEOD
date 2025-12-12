from setuptools import setup

package_name = 'opponent_tracker'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    install_requires=[
        'rclpy',
        'sensor_msgs',
        'nav_msgs',
        'geometry_msgs',
        'std_msgs',
        'visualization_msgs',
        'joblib',
        'numpy',
        'matplotlib',
        'pyyaml',
        'scipy',
    ],
    entry_points={
        'console_scripts': [
            'opponent_odom_node = opponent_tracker.opponent_odom_node:main',
            'evaluate_opponent = opponent_tracker.evaluate_opponent:main',
        ],
    },
)
