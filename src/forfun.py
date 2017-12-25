import theano
import numpy as np
from scipy.stats import mode
import theano.tensor as T
 
mask=T.fmatrix()
 
new_mask = T.switch(mask, mask, np.NINF)
 
 
 
test_model = theano.function([mask], T.nnet.softmax(new_mask))

import numpy as np

if __name__ == '__main__':


    X=np.asarray([[0.0,0.0,1.0,1.0,1.0,1.0],[0.0,0.0,0.0,1.0,1.0,1.0]], dtype='float32')
    print test_model(X)

