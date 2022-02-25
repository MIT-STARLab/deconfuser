# Deconfuser

Deconfuser is a library for fast orbit fitting to directly imaged multi-planetary systems:
* quickly fitting orbits to planet detections in 2D images
* guaranteeing that all orbits within a certain tolerance are found
* testing all groupings of detections by planets (which detection belongs to which planet)
* ranking partitions of detections by planets (deciding which assignment of detection-to-planet fits the data best)

## Installation

```bash
python setup.py install
```

## Usage

### Distributed Monte-Carlo scheme for estimating confusion rates
```bash
python test_deconfuser.py --help
python test_deconfuser.py 0.2 0.1 0.05 --n_planets 3 --n_epochs 3 --cadence 0.5 --min_a 0.5 --max_a 1.5 --n_processes 4 --n_systems 100 > results.txt
```

### Orbit fitting only example

```python
import numpy as np
import deconfuser.orbit_fitting as orbit_fitting
import deconfuser.sample_planets as sample_planets

#grid search for 3 equally spaced observations 4 months apart
mu = 4*np.pi**2 #in AU^3/year^2
ts = np.arange(0,1,1.0/3) #in years
of = orbit_fitting.OrbitGridSearch(mu, ts, max_e=0.3, min_a=0.8, max_a=1.2, tol=0.1)

#planet with a = 1, e = 0.3 and random orientation
a,e = 1,0.3
i,o,O,M0 = 2*np.pi*np.random.random(4)
xs,ys = sample_planets.get_observations(a, e, i, o, O, M0, ts, mu)
xys = np.concatenate([xs,ys]).T

#fit orbital elements and print RMS errors
err, (a, e, i, o, O, M0) = of.fit(xys)
xs_fit,ys_fit = sample_planets.get_observations(a, e, i, o, O, M0, ts, mu)
print(err, np.sqrt(np.mean((xs-xs_fit)**2 + (ys-ys_fit)**2)))
```


## License
[MIT](https://choosealicense.com/licenses/mit/)
