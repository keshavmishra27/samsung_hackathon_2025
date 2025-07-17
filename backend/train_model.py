from tensorflow.keras.applications import MobileNetV2
import pandas as pd 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, layers
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt

waste_arr=["Biodegradable","Ewaste","hazardous","Non Biodegradable","Pharmaceutical and Biomedical Waste"]

train_datagen = ImageDataGenerator(
    rescale=1/255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

valid_datagen = ImageDataGenerator(rescale=1/255)

train_dir = r"final_datset\train"
val_dir   = r"final_datset\val"

train_gen = train_datagen.flow_from_directory(
    train_dir, target_size=(224,224),
    batch_size=32, class_mode='sparse'
)

print(train_gen.class_indices)

valid_gen = valid_datagen.flow_from_directory(
    val_dir,   target_size=(224,224),
    batch_size=32, class_mode='sparse'
)


def build_model(optimizer='adam', learning_rate=1e-4, dropout_rate=0.3, dense_units=128, unfreeze_layers=50):
    base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

    # Freeze all layers except the last few
    base_model.trainable = True
    # Freeze all layers except the last `unfreeze_layers`
    for layer in base_model.layers[:-100]:  # Freeze more than before
        layer.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(dropout_rate),
        layers.Dense(dense_units, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(5, activation='softmax')  # 10 waste classes
    ])

    # Select optimizer
    if optimizer == 'adam':
        opt = optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        opt = optimizers.RMSprop(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        opt = optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")

    model.compile(
        optimizer=opt,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# Reset the generator
train_gen.reset()

final_model = build_model()

# Recompile for categorical data (since GridSearch used sparse categorical)
final_model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='rmsprop', 
    metrics=['accuracy']
)

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

earlystop_cb = EarlyStopping(patience=3, restore_best_weights=True)
checkpoint_cb = ModelCheckpoint("best_model.h5", save_best_only=True)
from tensorflow.keras.callbacks import ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.5)

# Reset generators
train_gen.reset()
valid_gen.reset()

history = final_model.fit(
    train_gen,
    steps_per_epoch = len(train_gen), 
    epochs=1,  # Use best epochs from grid search
    batch_size=32,  # Note: batch_size here won't override generator's batch_size
    validation_data=valid_gen,
    validation_steps=len(valid_gen) // 4,
    callbacks=[checkpoint_cb, earlystop_cb,reduce_lr]
)

# Plotting results
df = pd.DataFrame(history.history)
df[['loss','val_loss']].plot()
plt.title('Training and Validation Loss')
plt.show()

df[['accuracy','val_accuracy']].plot()
plt.title('Training and Validation Accuracy')
plt.show()
final_model.save(r'backend\models\garbage_tf_model.h5')