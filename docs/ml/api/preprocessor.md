## Defining a Preproccsor

Currently, there is no defined preprocessor interface. Instead it is expected for you to implement a Ray Preprocessor instead.

From there override the following in the Preprocessor class:
```python
def _fit(self, dataset: Dataset) -> Preprocessor:
```

and one of the following:
```python
def _transform_pandas(self, batch: pd.DataFrame) -> pd.DataFrame:

def _transform_numpy(self, np_data: np.ndarray | Dict[str, np.ndarray]) -> np.ndarray | Dict[str, np.ndarray]:
```
