from setuptools import setup, find_packages

setup(
    name="ear-job-visualization",
    version="5.2.0",
    author="Oriol Vidal, Jalal Lakhlili",
    author_email="oriol.vidal@eas4dc.com, jalal.lakhlili@eas4dc.com",
    description="High level support for read and visualize job information given by the EAR Library.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    license="EPL-2.0",
    keywords=["data", "visualization", "hpc", "analysis", "ear", "paraver"],
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={
        "": ["*", "*/*", "*/*/*"],  # This will include all files and subdirectories
    },
    include_package_data=True,
    install_requires=[
        "matplotlib",
        "pandas",
        "importlib_resources",
    ],
    entry_points={
        'console_scripts': [
            'ear-job-visualizer=ear_job_visualize.ear_job_viz:main',
        ],
    },
)
