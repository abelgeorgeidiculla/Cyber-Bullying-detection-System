from pycaret.classification import *

# Example: assuming your dataset is called df and target is 'label'
clf = setup(data=df, target='label', session_id=123)