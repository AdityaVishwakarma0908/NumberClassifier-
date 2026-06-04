import cv2
import numpy as np
import joblib

print("Loading model...")
model = joblib.load('mnist_model.pkl')
print("Model loaded!")

# 1. Back to a comfortable drawing size (280x280)
CANVAS_SIZE = 280
canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)
drawing = False

def draw(event, x, y, flags, param):
    global drawing
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        # 2. Increase brush radius back to 12 so we can see it!
        cv2.circle(canvas, (x, y), 12, (255), -1)
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(canvas, (x, y), 12, (255), -1)
            
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.circle(canvas, (x, y), 12, (255), -1)

window_name = 'Large Canvas (P: Predict, C: Clear, Q: Quit)'
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, draw)

print("\n--- Controls ---")
print("Draw with your mouse.")
print("Press 'P' to Predict.")
print("Press 'C' to Clear canvas.")
print("Press 'Q' to Quit.")

while True:
    cv2.imshow(window_name, canvas)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)
        print("Canvas cleared.")
    elif key == ord('p'):
        # 3. The magic step: Resize the large canvas down to 28x28 for the model
        resized_img = cv2.resize(canvas, (28, 28), interpolation=cv2.INTER_AREA)
        
        # Flatten it and predict
        flattened_img = resized_img.flatten()
        prediction = model.predict([flattened_img])
        print(f"\n---> The AI predicts you drew a: {prediction[0]} <---")

cv2.destroyAllWindows()