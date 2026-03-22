import cv2 
import time
cap = cv2.VideoCapture(0)

image_count = 0
while image_count < 20:
    success,frame = cap.read()

    if not success:
        print("There is Problem in Getting Frame")

    cv2.imshow("Frame",frame)

    cv2.imwrite(f"dataset/Eswararao/{image_count}.png",frame)
    image_count += 1
    print("Image Count : ",image_count)

    time.sleep(2)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

