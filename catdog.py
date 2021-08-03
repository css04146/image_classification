import pandas as pd
import numpy as np
from keras.preprocessing.image import *
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import os
import sys

import io

sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'utf-8')



path = "C:/Users/css04/OneDrive/바탕 화면/catdog/" # 데이터의 경로

# 데이터 형상 관련 상수 정의
IMAGE_WIDTH=128
IMAGE_HEIGHT=128
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNEL=3

# 학습 데이터 준비
filenames = os.listdir(path + "train")
# os.listdir() 메소드는 지정한 디렉토리 내의 모든 파일과 디렉토리의 리스트를 반환한다.
categories=[]
for filename in filenames:
    category=filename.split(".")[0] # category는 ‘.’을 기준으로 첫 번째로 정한다.
    if category =="dog": # ‘.’ 이전의 file 이름이 ‘dog’로 되어 있다면
        categories.append(1) # 리스트에 1을 넣는다.
    else: # category가 cat으로 되어 있다면
        categories.append(0) # 리스트에 0을 넣는다.
df = pd.DataFrame(
    {"filename":filenames, # filename은 filenames에서 가져옴
    "category":categories} # category와 filenames을 짝지은다.
)
df

sample = random.choice(filenames) # filenames에서 무작위로 한 개를 선정
image = load_img(path+"train/"+sample) # 선정된 이미지의 경로를 가져온다.
plt.imshow(image) # 가져온 이미지를 보여준다.
from keras.models import Sequential
# Sequential 모델은 레이러를 선형으로 연결하여 구성
# 레이어 인스턴스를 생성자에게 넘겨줌으로써 Sequential 모델을 구성할 수 있습니다.
from keras.layers import *
# import keras.layers와 문법상 동일
# keras.layers의 모든 모듈을 가져온다.
# from 모듈 import 메소드/변수

# 레이어 1
model = Sequential()
model.add(Conv2D(32, (3,3), activation="relu", input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH , IMAGE_CHANNEL)))
# add 메소드를 통해 Sequential 모델을 점진적으로 작성
# relu : 입력이 0 이하면 0으로 침묵, 0을 넘으면 입력 그대로를 출력하는 함수
model.add(BatchNormalization())
# 배치정규화
model.add(MaxPooling2D(pool_size=(2,2)))
# 풀링 이유 :
# 1. 이미지의 크기를 줄이면서 데이터의 손실을 막기 위해서
# 2. 합성곱 계층의 과적합을 막기 위해서 ( 견해에 따라 딥러닝 모델 자체가 과적합 하려고 하는 건데 방지한다는 것이 이상하다는 견해도 있음. )
model.add(Dropout(0.25))
# 노드를 학습에서 무시 ( 과적합 방지 )

# 레이어 2
model.add(Conv2D(64, (3,3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# 레이어3
model.add(Conv2D(128, (3,3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# Fully Connected
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(2,activation="softmax"))

# 모델 실행 옵션
model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=['accuracy'])
model.summary()

# reduceLROnPlateau
# : callback 함수의 일종, learning rate가 더이상 업데이트가 되지 않으면, 학습을 중단하여라
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

earlystop = EarlyStopping(patience=10)
learning_rate_reduction=ReduceLROnPlateau(
                        monitor= "val_accuracy",
                        patience = 2,
                        factor = 0.5,
                        min_lr=0.0001,
                        verbose=1)

callbacks = [earlystop, learning_rate_reduction]

# 이미지 제너레이터에서 class_mode = "categorical"로 지정하기 위해 컬럼 카테고리를 스트링으로 변경함.
df['category']=df['category'].replace({0:'cat',1:"dog"})

train_df, validate_df = train_test_split(df , test_size=0.2, random_state= 42)

train_df=train_df.reset_index(drop=True)
validate_df=validate_df.reset_index(drop=True)


train_df['category'].value_counts()
# >>>
# dog    10015
# cat     9985
# Name: category, dtype: int64

validate_df['category'].value_counts()
# >>>
# cat    2515
# dog    2485
# Name: category, dtype: int64

total_train=train_df.shape[0]
total_validate=validate_df.shape[0]
batch_size=15

# 트레이닝 데이터의 제너레이터 설정
train_datagen=ImageDataGenerator(
        rotation_range=15,
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1)

train_generator=train_datagen.flow_from_dataframe(
    train_df,
    path+"train",
    x_col = "filename",
    y_col = "category",
    target_size = IMAGE_SIZE,
    class_mode = "categorical",
    batch_size = batch_size )

validate_datagen=ImageDataGenerator(rescale=1./255)
# 검증이미지니까, 사진 그대로 쓰겠다.

validation_generator=validate_datagen.flow_from_dataframe(
    validate_df,
    path+"train",
    x_col= "filename",
    y_col= "category",
    target_size = IMAGE_SIZE,
    class_mode = "categorical",
    batch_size = batch_size )

example_df=train_df.sample(n=1).reset_index(drop=True)
example_df

example_generator = train_datagen.flow_from_dataframe(
                    example_df,
                    path+"train",
                    x_col = "filename",
                    y_col = "category",
                    target_size = IMAGE_SIZE,
                    class_mode = "categorical")

plt.figure(figsize=(10,10))
for i in range(0,15):
    plt.subplot(5,3,i+1)
    for xBatch, yBatch in example_generator:
        image = xBatch[0]
        plt.imshow(image)
        break
plt.tight_layout()
plt.show()

epochs = 3

history = model.fit_generator(
    train_generator,
    epochs = epochs,
    steps_per_epoch = total_train//batch_size ,
    validation_data=  validation_generator,
    validation_steps = total_validate//batch_size,
    callbacks = callbacks,
)

# 모델 저장
model.save_weights("model.h5")

historyDict=history.history

acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

epo = range(1, len(acc)+1)
plt.plot(epo, loss, 'bo', label="Traing loss")
plt.plot(epo, val_loss, 'b', label="Val loss")
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.plot(epo, acc, 'ro', label="Traing accuracy")
plt.plot(epo, val_acc, 'r', label="Val accuracy")
plt.xlabel("epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

plt.plot(epo, acc, 'ro', label="Traing accuracy")
plt.plot(epo, val_acc, 'r', label="Val accuracy")
plt.xlabel("epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

test_datagen=ImageDataGenerator(rescale=1./255)
# 테스트 이미지니까, 사진 그대로 씀
test_generator=test_datagen.flow_from_dataframe(
    test_df,
    path+"test",
    x_col= "filename",
    y_col= None,
    target_size = IMAGE_SIZE,
    class_mode = None,
    batch_size = batch_size,
    shuffle = False)

# 3. 예측
predict=model.predict_generator(test_generator,
                                steps=nbsamples/batch_size,
                                callbacks=callbacks)

test_df['category']=np.argmax(predict, axis=1)

test_df['category']=test_df['category'].replace({0:'cat',1:"dog"})
ex_df=test_df.sample(n=1).reset_index(drop=True)
ex_df

ex_generator = test_datagen.flow_from_dataframe(
                    ex_df,
                    path+"test",
                    x_col = "filename",
                    y_col = None,
                    target_size = IMAGE_SIZE,
                    class_mode = None)
test_sample=list(ex_df.filename)

sample = ""
for test in test_sample:
    sample += test
image = load_img(path+"test/"+sample)
plt.figure(figsize=(8,8))
plt.imshow(image)

plt.tight_layout()
plt.show()

sampleSubmission=pd.read_csv(path+"sampleSubmission.csv", dtype="object")
sampleSubmission

index=[]
for filename in test_df.filename:
    li=filename.split(".")[0]
    index.append(li)

test_df["id"]=index

final=test_df.merge(sampleSubmission)[['id','category']]
final['id']=final['id'].astype("int64")
final=final.sort_values("id")

final.rename({'category':"label"},axis='columns').to_csv("Submission.csv", index=False)
