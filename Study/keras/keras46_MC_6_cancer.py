from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

datasets=load_breast_cancer()
x=datasets.data
y=datasets.target

y=y.reshape(-1, 1)

x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=0.8, random_state=11)
x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, train_size=0.8, random_state=11)

scaler=MinMaxScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
x_val=scaler.transform(x_val)

enc=OneHotEncoder()
enc.fit(y_train)
y_train=enc.transform(y_train).toarray()
y_test=enc.transform(y_test).toarray()
y_val=enc.transform(y_val).toarray()

input1=Input(shape=(x_train.shape[1]))
dense1=Dense(150, activation='relu')(input1)
dense1=Dense(200, activation='relu')(dense1)
dense1=Dense(150, activation='relu')(dense1)
dense1=Dense(100, activation='relu')(dense1)
dense1=Dense(80, activation='relu')(dense1)
dense1=Dense(60, activation='relu')(dense1)
output1=Dense(2, activation='sigmoid')(dense1)
model=Model(input1, output1)

cp=ModelCheckpoint(filepath='./skanwngud/Study/modelCheckpoint/k46_6_cancer_{epoch:02d}-{val_loss:.4f}.hdf5',
                    monitor='val_loss', save_best_only=True, mode='auto')
early=EarlyStopping(monitor='val_loss', patience=10, mode='auto')

model.compile(loss='binary_crossentropy', optimizer='adam', metrics='acc')
model.fit(x_train, y_train, epochs=100, batch_size=8, validation_data=(x_val, y_val), callbacks=[early, cp])

loss=model.evaluate(x_test, y_test)
pred=model.predict(x_test)

print('loss : ', loss[0])
print('acc : ', loss[1])

# results
# loss :  0.07478024810552597
# acc :  0.9649122953414917