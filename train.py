from utils.data_preprocessing import load_data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.optimizers import Adam


def build_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(4, activation='softmax')  # Assuming 4 categories
    ])
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def main():
    train_images, train_labels, val_images, val_labels = load_data()
    input_shape = (train_images.shape[1], train_images.shape[2], train_images.shape[3])  # Adjust shape as needed
    model = build_model(input_shape)

    # Train the model
    model.fit(train_images, train_labels, epochs=10, validation_data=(val_images, val_labels))

    # Save the model
    model.save('models/mood_detection_model.h5')


if __name__ == "__main__":
    main()
