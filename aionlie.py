#!/usr/bin/env python
# coding: utf-8

# In[40]:


pip install nbconvert


# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import os


# In[4]:


df = pd.read_csv('/Users/lyw/baseline/train/train.csv')#,index=False)
df


# In[5]:


# 옥외누수(out), 옥내누수(in), 정상(normal), 전기/기계음(noise), 환경음(other)

df['class']=0
df.loc[df['leaktype'] == 'out','class']=0
df.loc[df['leaktype'] == 'in','class']=1
df.loc[df['leaktype'] == 'normal','class']=2
df.loc[df['leaktype'] == 'noise','class']=3
df.loc[df['leaktype'] == 'other','class']=4
df


# In[6]:


x = df[df.columns[1:514].to_list()]
x


# In[7]:


print(type(x))


# In[8]:


np.eye(5)[3]


# In[9]:


y = df[df.columns[-1:].to_list()]
y


# In[10]:


yy = y['class'].map(lambda x:np.eye(5)[x])


# In[11]:


yy = np.vstack(yy)


# In[12]:


print(type(yy))


# In[13]:


np.shape(yy)


# In[14]:


np.shape(x)


# In[15]:


x.shape


# In[16]:


[x.shape, np.newaxis]


# In[17]:


x = np.expand_dims(x, axis=-1)


# In[18]:


import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM,Dense, Dropout, Activation
#from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping



# x = df[df.columns[1:].to_list()]
# y = df[df.columns[-1:].to_list()]

# train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, yy, test_size=0.2, shuffle=True, stratify=None, random_state=100)


#train, val = train_test_split(df, test_size=0.2, random_state=100)

#print(train.shape[0])

model = tf.keras.models.Sequential()

# model.add(tf.keras.Input(shape=x_train.shape[1:]))
# model.add(LSTM(units = 50, return_sequences = True))
# model.add(Dropout(0.2))
        
# model.add(LSTM(units = 50, return_sequences = True))
# model.add(Dropout(0.2))

# model.add(LSTM(units=50,return_sequences = True))
# model.add(Dropout(0.2))

# model.add(LSTM(units=50))
# model.add(Dropout(0.2))
model.add(tf.keras.Input(shape=x_train.shape[1:]))
model.add(LSTM(units = 50, return_sequences = True))

model.add(LSTM(units = 50, return_sequences = True))

model.add(LSTM(units=50))

model.add(Dense(units = 5,activation='softmax'))
model.summary()


#print(train)
#print(val)


# In[19]:


filepath="/Users/lyw/baseline/train/weights-improvement-{epoch:02d}.ckpt"#-{val_acc:.2f}
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=2, save_best_only=True, mode='min')
callbacks_list = [checkpoint]


# In[20]:


model.compile(optimizer = 'adam', loss = 'categorical_crossentropy',metrics = ['accuracy']) #'sparse_categorical_crossentropy' categorical_crossentropy
filepaths = '/Users/lyw/baseline/train/best_model6_.h5'
model.load_weights(filepaths)
# early_stopping = EarlyStopping() # 조기종료 콜백함수 정의
history = model.fit(x_train,
                    y_train,
                    batch_size=16,
                    epochs=1,
                    validation_steps=5,
                    validation_data=(x_test, y_test),
                    validation_batch_size=16,
                    verbose=2,
                    callbacks=callbacks_list)


# In[ ]:


#from tensorflow.keras.models import save_model #학습 모델 저장


# In[ ]:


#save_model(model, 'best_model.h5')


# In[21]:


filepath = '/Users/lyw/baseline/train/best_model6_.h5'
model.load_weights(filepath)# Re-evaluate the model
loss,acc = model.evaluate(x_test, y_test)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))


# In[22]:


print(x_test.shape)


# In[23]:


y_test.shape


# In[24]:


from sklearn.metrics import confusion_matrix,accuracy_score, recall_score, precision_score, f1_score, matthews_corrcoef

#print(x_test)
pred = model.predict(x_test)
# cm = confusion_matrix(y_test,pred)
# df_cm = pd.DataFrame(cm,index=['normal', 'abnormal'],columns=['normal', 'abnormal'])
# df_cm


# In[25]:


pred


# In[26]:


pred_arg = np.argmax(pred, axis=1)


# In[27]:


pred_arg


# In[28]:


true_arg = np.argmax(y_test, axis=1)


# In[29]:


true_arg


# In[30]:


cm = confusion_matrix(true_arg, pred_arg)
print(cm)


# In[31]:


f1_score(true_arg, pred_arg, average='macro')


# In[32]:


accuracy_score(true_arg, pred_arg)


# In[ ]:





# In[ ]:


print('F1_score:{:4.2f}%'.format(f1_score(true_arg, pred_arg, average=None)*100))


# In[ ]:


test_df = pd.read_csv('/Users/lyw/baseline/test/test.csv')
test_X = test_df.loc[:,test_df.columns!='id']

#print(test_X)

test_ids = test_df['id']
test_X = np.expand_dims(test_X, axis=-1)

#print(test_X)
#print(type(test_X))
    
model = filepath
#print(f'complete {train_serial} model load')
    
print('Making predictions')
    
sample_df = pd.read_csv('/Users/lyw/baseline/sample_submission.csv')
sorter = list(sample_df['id'])
    
y_pred = model.predict(test_X)
print(type(y_pred))
y_pred_df = pd.DataFrame(y_pred, columns=['leaktype'])
y_pred_df['leaktype'] = y_pred_df['leaktype'].replace(LABEL_DECODING)
pred_df = pd.concat([test_ids, y_pred_df],axis=1)
    
    # sort predictions
resdf = pred_df.set_index('id')
result = resdf.loc[sorter].reset_index()
resultpath = os.path.join('/Users/lyw/baseline/predictions.csv')
result.to_csv(resultpath, index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


import keras.backend as K


# In[ ]:


model.evaluate(x_test, y_test)


# In[ ]:


import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']
acc = history.history['acc']
val_acc = history.history['val_acc']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, acc, 'ro', label='Training acc')
plt.plot(epochs, val_acc, 'bo', label='Validation acc')
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()


# In[ ]:


loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss)+1)
plt.figure()
plt.plot(epochs, loss, label='Training loss')
plt.plot(epochs, val_loss, label='Validation loss')
plt.title('Training and Valdation loss')
plt.legend()
plt.show()  


# In[37]:


get_ipython().system('pip install matplotlib')


# In[38]:


import matplotlib.pyplot as plt
plt.plot(x_test[0]) # in -> out


# In[39]:


plt.plot(x_test[100])


# In[ ]:


c


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




