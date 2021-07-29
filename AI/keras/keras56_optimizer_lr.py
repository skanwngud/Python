import numpy as np

# 1. Data
x=np.array([1,2,3,4,5,6,7,8,9,10])
y=np.array([1,2,3,4,5,6,7,8,9,10])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 2. Modeling
model=Sequential()
model.add(Dense(1000, input_dim=1))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))

# 3. Compile, Fitting
from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam

optimizer=Adam(lr=0.001)
# optimizer=Adadelta(lr=0.0001)
# optimizer=Adamax(lr=0.0001)
# optimizer=Adagrad(lr=0.0001)
# optimizer=RMSprop(lr=0.0001)
# optimizer=SGD(lr=0.0001)
# optimizer=Nadam(lr=0.0001)

model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
model.fit(x,y, epochs=100, batch_size=1)

# 4. Evaluate, predict
loss, mse=model.evaluate(x,y, batch_size=1)
y_pred=model.predict([11])
print('loss : ', loss, 'results', y_pred)

# results - Adam(lr=0.001)
# loss :  1.3599787528767449e-12 results [[10.999997]]

# results - Adam(lr=0.01)
# loss :  0.0005418954533524811 results [[10.982619]]

# results - Adam(lr=0.1)
# loss :  0.04781406372785568 results [[11.399299]]

# results - Adam(lr=0.0001)
# loss :  5.977728051220765e-06 results [[10.995426]]

# results - Adadelta(lr=0.1)
# loss :  1.6021660940168658e-06 results [[10.997427]]

# results - Adadelta(lr=0.01)
# loss :  0.0008832619641907513 results [[10.94282]]

# results - Adadelta(lr=0.001)
# loss :  15.045745849609375 results [[4.065893]]

# results - Adadelta(lr=0.0001)
# loss :  36.16326141357422 results [[0.33031717]]

# results - Adamax(lr=0.1)
# loss :  2.8322644233703613 results [[8.596212]]

# results - Adamax(lr=0.01)
# loss :  3.439026949499102e-13 results [[10.999995]]

# results - Adamax(lr=0.001)
# loss :  3.8053826756367926e-06 results [[10.996331]]

# results - Adamax(lr=0.0001)
# loss :  0.0037116818130016327 results [[10.929584]]

# results - Adagrad(lr=0.1)
# loss :  2.132985830307007 results [[12.921034]]

# results - Adagrad(lr=0.01)
# loss :  2.614295340208628e-07 results [[11.000804]]

# results - Adagrad(lr=0.001)
# loss :  9.557895282341633e-06 results [[10.999047]]

# results - Adagrad(lr=0.0001)
# loss :  0.0046218885108828545 results [[10.914852]]

# results - RMSprop(lr=0.1)
# loss :  95411640.0 results [[21010.79]]

# results - RMSprop(lr=0.01)
# loss :  6.050080299377441 results [[5.822253]]

# results - RMSprop(lr=0.001)
# loss :  0.001002247678115964 results [[11.068377]]

# results - RMSprop(lr=0.0001)
# loss :  0.00026124255964532495 results [[10.966665]]

# results - SGD(lr=0.1)
# nan

# results - SGD(lr=0.01)
# nan

# results - SGD(lr=0.001)
# loss :  8.566111318941694e-06 results [[10.997222]]

# results - SGD(lr=0.0001)
# loss :  0.0015608479734510183 results [[10.956277]]

# results - Nadam(lr=0.1)
# loss :  5.2560609198337715e-09 results [[11.000098]]

# results - Nadam(lr=0.01)
# loss :  235690.875 results [[-150.5253]]

# results - Nadam(lr=0.001)
# loss :  5.202139163884567e-09 results [[11.000109]]

# results - Nadam(lr=0.0001)
# loss :  2.1915486286161467e-05 results [[10.995763]]