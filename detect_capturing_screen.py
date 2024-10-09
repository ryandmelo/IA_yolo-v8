from ultralytics import YOLO
import cv2
from windowcapture import WindowCapture
from collections import defaultdict
import numpy as np

#wincap = WindowCapture("Nome_da_Janela")
offset_x = 400 #0
offset_y = 300 #30
wincap = WindowCapture(size=(1280, 720), origin=(0, 0))

# Usa modelo da Yolo

model = YOLO("yolov8n.pt")

track_history = defaultdict(lambda: [])
seguir = True
deixar_rastro = True

while True:
    img = wincap.get_screenshot()

    if seguir:
        results = model.track(img, persist=True)
    else:
        results = model(img)

    # Process results list
    for result in results:
        # Visualize the results on the frame
        img = result.plot()

        if seguir and deixar_rastro:
            try:
                # Get the boxes and track IDs
                boxes = result.boxes.xywh.cpu()
                track_ids = result.boxes.id.int().cpu().tolist()

                # Plot the tracks
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y)))  # x, y center point
                    if len(track) > 30:  # retain 90 tracks for 90 frames
                        track.pop(0)

                    # Draw the tracking lines
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(img, [points], isClosed=False, color=(230, 0, 0), thickness=5)
            except:
                pass

    cv2.imshow("Tela", img)

    k = cv2.waitKey(1)
    if k == ord('q'):
        break

cv2.destroyAllWindows()
print("desligando")