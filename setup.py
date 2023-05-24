import os

from setuptools import setup, find_packages
from setuptools.command.install import install

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

class CustomInstall(install):
    def run(self):
        install.run(self)
        os.system("pip install -r requirements.txt")

setup(
    name="pastmaker",
    version="0.1",
    description="Shift text of present or future tense to past tense.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Sophie Codes",
    url="https://github.com/sophiecodes42/pastmaker",
    packages=find_packages(),
    install_requires=["spacy>=3.5.0", "pattern", "regex",
                        "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.5.0/en_core_web_sm-3.5.0-py3-none-any.whl",
                        "en_core_web_trf @ https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.5.0/en_core_web_trf-3.5.0-py3-none-any.whl"
                        ],
    cmdclass={"install": CustomInstall},
      )