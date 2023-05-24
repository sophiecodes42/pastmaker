# pastmaker
pastmaker is based on tenseflow by @bendichter and automatically changes the tense of any English text from present and future tense to past tense


## Features
- change text from `'present'`- or `'future'` tense to `'past'` tense

## Installation

Install this package
```
git clone https://github.com/sophiecodes42/pastmaker.git
cd pastmaker
pip install .
```
Install requirements
```
pip install -r requirements.txt
```

## Usage
Basic usage
```python
from pastmaker import pastmaker

pastmaker('I will go to the store.')
u'I went to the store.'
```

