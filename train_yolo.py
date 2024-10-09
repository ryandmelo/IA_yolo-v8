from ultralytics import YOLO

# para marcar as imagens
# https://www.makesense.ai/

def main():
    # Load a model
    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

    # Use the model
    model.train(data="data.yaml", epochs=5, batch=-1)  # train the model
    metrics = model.val()  # evaluate model performance on the validation set
    # results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image


if __name__ == '__main__':
    # freeze_support()
    main()
