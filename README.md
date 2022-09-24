# Tamader
### Generate sample data that matches the given statistical distribution and conditions

## Usage
- Check if given conditions are reasonable.
- hmm... generate fake data?

## Currently supported distributions
- Normal distribution: normal

## How to use

### Use it as a module
```python3
import scripts.common as common
from scripts.tamader import Tamader

# Set Logger
config = common.get_config(test=True)
logger = common.get_logger(__name__)
common.set_logger_level(logger, config['logging_level'])

# Initialize an instance
agent = Tamader(logger=logger)

# Get results
outcome = agent.process(
    distribution="normal",
    mean=1.0,
    std=4.0,
    sample_size=5,
    boundary=[-4, float("inf")]
)
# outcome = [-4., 5.95517944, 0.69340047, 3.84876502, -1.4973449 ] for example

# If the condition is not reasonable
outcome = agent.process(
    distribution="normal",
    mean=1.0,
    std=4.0,
    sample_size=30,
    boundary=[-4, 2]
)
# outcome = None
```

### Use API

- Start the server using command line
```!#/bin/bash
python app.py
```
- Curl!
```!#/bin/bash
curl localhost:5000/calculate -X POST -H 'Content-Type: application/json' -d @'example/input.json' | python -m json.tool
```

### TODO

- Add test cases
- Enhance document
- Add new distributions
