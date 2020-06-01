
import CT_regression_tools
import numpy as np

#model = CT_regression_tools.modelcreation1(2, 4, 20)
#model.save('model4.h5')
model = CT_regression_tools.load_model('model4.h5') #load model from h5 file

X_test, Y_test = CT_regression_tools.loadfromh5(2, 3, 3) #testing data
X_test = np.asarray(X_test).reshape(np.asarray(X_test).shape[0], 64, 64, 1)

CT_regression_tools.eval(model, X_test, Y_test)

