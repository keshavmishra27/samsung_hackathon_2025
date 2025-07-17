import cv2
import numpy as np

def universal_object_counter(image_path, show_result=True, output_path=None):
    image = cv2.imread(image_path)
    if image is None:
        print("Image not found or path is incorrect.")
        return 0

    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 1: Smooth to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step 2: Adaptive Threshold to cover uneven lighting
    adaptive = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 7
    )

    # Step 3: Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Step 4: Combine both binary + edges
    combined = cv2.bitwise_or(adaptive, edges)

    # Step 5: Morphology to close gaps
    kernel = np.ones((5, 5), np.uint8)
    morphed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

    # Step 6: Contour detection
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = h / float(w)

            # General aspect ratio filter (can be tuned)
            if 0.3 < aspect_ratio < 6:
                cv2.rectangle(original, (x, y), (x + w, y + h), (0, 255, 0), 2)
                count += 1

    print(f"Total objects found: {count}")

    if show_result:
        cv2.imshow("Detected Objects", original)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if output_path:
        cv2.imwrite(output_path, original)
        print(f"Saved detected image at: {output_path}")

    return count