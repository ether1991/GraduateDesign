

import numpy as np
y = np.array([1,2,3,4,5,8])

y = np.array(y, dtype='int').ravel()
print np.max(y)