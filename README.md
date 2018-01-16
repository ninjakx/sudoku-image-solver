# sudoku-image-solver

## GETTING STARTED:

- Clone the repo - `git clone git@github.com:ninjakx/sudoku-image-solver.git` and cd into any directory.

- Create a virtual environment with Python 3 and install dependencies.

```
$ virtualenv sudoku --python=/path/to/python3
$ source sudoku/bin/activate
```

- Run `sudoku.py`

## Command :
e.g:
```
$ python sudoku.py --preprocess 1 --image images/s2.jpg

```
### preprocess 1 for transforming image
### preprocess 2 for already transformed image



references:

[grid points](https://stackoverflow.com/questions/10196198/how-to-remove-convexity-defects-in-a-sudoku-square)

[get features (of digits)](https://medium.com/@neshpatel/solving-sudoku-part-ii-9a7019d196a2)

[Images samples](https://github.com/eatonk/sudoku-image-solver/tree/master/sudoku_images)
