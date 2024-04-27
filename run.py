import cv2
import pyautogui
import numpy as np
from model import Model

def detect_violence(frame, model):
    # Perform violence detection on the frame
    label = model.predict(image=frame)['label']
    return label

if __name__ == '__main__':
    model = Model()
    
    # Define the screen region you want to capture
    screen_region = (100, 80, 1000, 900)  # Adjust this to the region you want to capture
    
    while True:
        # Capture screen
        screen = pyautogui.screenshot(region=screen_region)
        
        # Convert PIL image to OpenCV format
        frame = cv2.cvtColor(np.array(screen), cv2.COLOR_RGB2BGR)
        
        # Perform violence detection
        label = detect_violence(frame, model)
        
        # Draw white background for the text
        text_size, _ = cv2.getTextSize(label.title(), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        text_x, text_y = 50, 50  # Position of the text
        cv2.rectangle(frame, (text_x, text_y - text_size[1]), (text_x + text_size[0], text_y), (265, 265, 265), -1)
        
        # Draw text on the white background
        cv2.putText(frame, label.title(), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Display the frame with label
        cv2.imshow('Violence Detection', frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
