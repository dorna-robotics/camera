import setuptools
import camera as pkg

# Read the version from the dorna_vision package
version = pkg.__version__


with open("README.md", "r") as fh:
    readme = fh.read()

setuptools.setup(
    name="camera",
    version= version,
    author="Dorna Robotics",
    author_email="info@dorna.ai",
    description="Python API for Intel RealSense camera and Dorna 2 robotic arm.",
    long_description=readme,
    long_description_content_type='text/markdown',
    url="https://dorna.ai/",
    project_urls={
        'gitHub': 'https://github.com/dorna-robotics/camera',
    },
    package_data={
        'camera': ['preset/*'],
    },     
    packages=setuptools.find_packages(),
    classifiers=[
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3.7',
        "Operating System :: OS Independent",
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    license="MIT",
    include_package_data=True,
    zip_safe = False,
)