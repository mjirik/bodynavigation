

import CT_regression_tools
import numpy as np

#model = CT_regression_tools.modelcreation2()
#model.save('model5.h5')
model = CT_regression_tools.load_model('model5.h5') #load model from h5 file

X_test, Y_test = CT_regression_tools.loadfromh5(1, 13, 13) #testing data
X_test = np.asarray(X_test).reshape(np.asarray(X_test).shape[0], 64, 64, 1)

CT_regression_tools.eval(model, X_test, Y_test)

