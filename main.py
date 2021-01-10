import os
import shutil
from PIL import Image, ImageOps  
# import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
import plotly.graph_objects as go
import plotly.express as px
import datetime
from plotly.offline import init_notebook_mode, iplot
import plotly.io as pio
pio.templates.default = 'plotly_white'
from plotly.subplots import make_subplots
init_notebook_mode(connected=True)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Concatenate, Flatten, MaxPooling2D, Conv2D
from  tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_addons as tfa
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import metrics
import efficientnet.tfkeras as efn
from tqdm import tqdm_notebook

# Reading through the metadata
summary = pd.read_csv('archive/coronahack-chest-xraydataset/Chest_xray_Corona_dataset_Summary.csv')
df = pd.read_csv('archive/coronahack-chest-xraydataset/Chest_xray_Corona_Metadata.csv')
print(df.head())

replace_dict = {'Pnemonia':1,
                'Normal':0}
df['Label'] = df['Label'].replace(replace_dict)

train_df = df[df.Dataset_type=='TRAIN']
print(train_df.head())
test_df = df[df.Dataset_type=='TEST']
print(test_df.head())


# Inside the Pneumonia idagnosed data how many are covid positive
df_pneumonia = df[df.Label==1]
df_pneumonia_covid = df_pneumonia[df_pneumonia.Label_2_Virus_category=='COVID-19']
pneumonia_covid_images = df_pneumonia_covid.X_ray_image_name.values.tolist()

# Defining the path to Train and Test directories
training_data_path = 'archive/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train'
testing_data_path = 'archive/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/test'

# Funtions for Making nd Removing subdirectories
def create_dir():
    try:
            os.makedirs('working/train/Pneumonia')
            os.makedirs('working/train/Normal')
            os.makedirs('working/test/Pneumonia')
            os.makedirs('working/test/Normal')
    except:
            pass

def remove_dir():
    try:
            shutil.rmtree('working/train')
            shutil.rmtree('working/test')    
    except:
            pass


# Seperate dataframes for different labels in test and train
train_pneumonia_df = train_df[train_df.Label==1]
train_normal_df = train_df[train_df.Label==0]
test_pneumonia_df = test_df[test_df.Label==1]
test_normal_df = test_df[test_df.Label==0]

# Copying the files to newly created locations. You may use Flow from dataframe attribute and skip all these steps. But I prefer to use flow from directory 
remove_dir()
create_dir()

training_images_pneumonia = train_pneumonia_df.X_ray_image_name.values.tolist()
training_images_normal = train_normal_df.X_ray_image_name.values.tolist()
testing_images_pneumonia = test_pneumonia_df.X_ray_image_name.values.tolist()
testing_images_normal = test_normal_df.X_ray_image_name.values.tolist()

for image in training_images_pneumonia:
        train_image_pneumonia = os.path.join(training_data_path, str(image))
        shutil.copy(train_image_pneumonia, 'working/train/Pneumonia')
                
for image in training_images_normal:
        train_image_normal = os.path.join(training_data_path, str(image))
        shutil.copy(train_image_normal, 'working/train/Normal')
                
for image in testing_images_pneumonia:
        test_image_pneumonia = os.path.join(testing_data_path, str(image))
        shutil.copy(test_image_pneumonia, 'working/test/Pneumonia')
                
for image in testing_images_normal:
        test_image_normal = os.path.join(testing_data_path, str(image))
        shutil.copy(test_image_normal, 'working/test/Normal')


# Model configuration
batch_size = 64
img_width, img_height, img_num_channels = 224,224,3
no_epochs = 15
verbosity = 1
input_shape = (img_width, img_height, img_num_channels)

#Creating an EffNet model
model_B7 = efn.EfficientNetB7(weights='imagenet', input_shape=input_shape, include_top=False)

# Function to build, compile and train the model

train_datagen = ImageDataGenerator(rescale=1./255,
shear_range=0.2,
rotation_range=0.2,
width_shift_range=0.2,
height_shift_range=0.2,
zoom_range=0.2,
horizontal_flip=True,
vertical_flip=True,
fill_mode='nearest')

train_generator = train_datagen.flow_from_directory('working/train',
target_size=(224,224),
batch_size=batch_size,
class_mode='binary')

valid_datagen = ImageDataGenerator(rescale=1./255)
valid_generator = valid_datagen.flow_from_directory('working/test',
target_size=(224,224),
batch_size=batch_size,
class_mode='binary')

pretrained_model = model_B7
pretrained_model.trainable=True
set_trainable=False

for layer in pretrained_model.layers:
        if layer.name == 'block7c_project_conv':
                set_trainable=True
        if set_trainable:
                layer.trainable=True
        else:
                layer.trainable=False


model=Sequential()
model.add(pretrained_model)
model.add(MaxPooling2D(name="MaxPool_"))
model.add(Dropout(0.2, name="dropout_out"))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.compile(loss=binary_crossentropy,
optimizer=Adam(),
metrics=[metrics.AUC(name='auc'), 'accuracy'])
es_callback = EarlyStopping(monitor='val_auc', mode='max', patience=8,
verbose=1, min_delta=0.0001, restore_best_weights=True)

history = model.fit(train_generator,
steps_per_epoch=train_generator.samples//batch_size,
epochs = no_epochs,
validation_data=valid_generator,
validation_steps=valid_generator.samples//batch_size,
callbacks= [es_callback],
verbose=verbosity)


#Plotting the evaluation metrics
fig = go.Figure()
fig.add_trace(go.Scatter(x=list(range(1,11)), y=history.history['auc'],
                         line=dict(color='firebrick', width=2, dash='dash'), name='AUC'))
fig.add_trace(go.Scatter(x=list(range(1,11)), y=history.history['val_auc'],
                         line=dict(color='turquoise', width=2), name='validation AUC'))

fig.add_trace(go.Scatter(x=list(range(1,11)), y=history.history['accuracy'],
                         line=dict(color='orange', width=2, dash='dash'), name='accuracy'))
fig.add_trace(go.Scatter(x=list(range(1,11)), y=history.history['val_accuracy'],
                         line=dict(color='green', width=2), name='validation accuracy'))

fig.update_layout(title_text='Plot of evaluation metrics', font_size=15, xaxis_title='Epochs')
fig.show()
