import pandas as pd
import numpy as np
import pickle
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import EfficientNetB0
from keras.models import Model
from keras.layers import GlobalAveragePooling2D
from keras.optimizers import Adam
from math import ceil

df = pd.read_csv("styles.csv", error_bad_lines=False)
df['image'] = df.apply(lambda row: str(row['id']) + ".jpg", axis=1)
df = df.sample(frac=1).reset_index(drop=True)

batch_size = 64  # Reduced for better performance

def train_model(label, model_name):
    image_generator = ImageDataGenerator(validation_split=0.2)
    
    training_generator = image_generator.flow_from_dataframe(
        dataframe=df,
        directory="images",
        x_col="image",
        y_col=label,
        target_size=(224, 224),
        batch_size=batch_size,
        subset="training"
    )
    
    validation_generator = image_generator.flow_from_dataframe(
        dataframe=df,
        directory="images",
        x_col="image",
        y_col=label,
        target_size=(224, 224),
        batch_size=batch_size,
        subset="validation"
    )
    
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers:
        layer.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    predictions = Dense(len(training_generator.class_indices), activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=['accuracy'])
    
    model.fit_generator(
        generator=training_generator,
        steps_per_epoch=ceil(0.8 * (df.shape[0] / batch_size)),
        validation_data=validation_generator,
        validation_steps=ceil(0.2 * (df.shape[0] / batch_size)),
        epochs=5,
        verbose=1
    )
    
    model.save(model_name + ".h5")
    
    class_indices = training_generator.class_indices
    pickle.dump(list(class_indices.keys()), open(model_name + '_keys.pkl', 'wb'))
    pickle.dump(list(class_indices.values()), open(model_name + '_vals.pkl', 'wb'))
    
    print(f"Model {model_name}.h5 saved!")

# Train models for all attributes
task_labels = ["masterCategory", "subCategory", "season", "gender"]
model_names = ["model_category", "model_subcategory", "model_season", "model_gender"]

for label, model_name in zip(task_labels, model_names):
    train_model(label, model_name)
