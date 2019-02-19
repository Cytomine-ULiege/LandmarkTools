# -*- coding: utf-8 -*-

# * Copyright (c) 2018.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *      http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.

from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='LandmarkTools',
    version='0.0.2b',
    description='Tools for the Landmark Detection algorithms.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=['ldmtools'],
    url='http://uliege.cytomine.org',
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved',
        'Programming Language :: Python',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Operating System :: OS Independent'
    ],
    install_requires=['numpy', 'scipy', 'pillow', 'joblib', 'imageio'],
    license='LICENSE',
)
