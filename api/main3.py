import cv2
import dlib
import numpy as np
import os
import sys


# Paths for required resources
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICTOR_PATH = os.path.join(BASE_DIR, "data", "shape_predictor_68_face_landmarks.dat")
FRAMES_DIR = os.path.join(BASE_DIR, "frame")

# Load Dlib's face detector and shape predictor
try:
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
except Exception as e:
    print(f"Error loading Dlib resources: {e}")
    sys.exit(1)
try:
    frames = [
        cv2.imread(os.path.join(FRAMES_DIR, f), cv2.IMREAD_UNCHANGED)
        for f in os.listdir(FRAMES_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    if not frames:
        raise FileNotFoundError("No valid frames found in the 'frame/' directory.")
except Exception as e:
    print(f"Error loading frames: {e}")
    sys.exit(1)


def resize_and_rotate_frame(frame, landmarks, scale_factor=1.0, rotation_angle=0):
    """Resize and rotate the frame based on face landmarks."""
    frame_width, frame_height = frame.shape[1], frame.shape[0]

    # Calculate face width (distance between the leftmost and rightmost landmarks)
    face_width = np.linalg.norm(landmarks[0] - landmarks[16])

    # Scale frame to match face width
    scale_factor *= face_width / frame_width
    new_width = int(frame_width * scale_factor)
    new_height = int(frame_height * scale_factor)
    resized_frame = cv2.resize(
        frame, (new_width, new_height), interpolation=cv2.INTER_AREA
    )

    # Rotate the frame
    center = (new_width // 2, new_height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    rotated_frame = cv2.warpAffine(
        resized_frame, rotation_matrix, (new_width, new_height), flags=cv2.INTER_LINEAR
    )

    # Calculate position (align to eye level)
    left_eye = np.mean(landmarks[36:42], axis=0)
    right_eye = np.mean(landmarks[42:48], axis=0)
    eye_center_x = int((left_eye[0] + right_eye[0]) / 2)
    eye_center_y = int((left_eye[1] + right_eye[1]) / 2)
    y_offset = int(new_height * 0.5)
    x = eye_center_x - new_width // 2
    y = eye_center_y - y_offset

    return rotated_frame, (x, y)


def overlay_frame(image, frame, position, opacity=1.0):
    """Overlay the virtual frame on the webcam image."""
    x, y = position
    h, w = frame.shape[:2]

   # Crop frame if it goes out of bounds
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(x + w, image.shape[1]), min(y + h, image.shape[0])

    frame_cropped = frame[y1 - y:y2 - y, x1 - x:x2 - x]
    if frame_cropped.shape[2] == 4:  # RGBA
        alpha = frame_cropped[:, :, 3] / 255.0 * opacity
        for c in range(3):
            image[y1:y2, x1:x2, c] = (
                alpha * frame_cropped[:, :, c]
                + (1 - alpha) * image[y1:y2, x1:x2, c]
            )
    # Ensure the frame is within bounds
    if y < 0 or x < 0 or y + h > image.shape[0] or x + w > image.shape[1]:
        return  # Skip if out of bounds

    if frame.shape[2] == 4:  # RGBA (with transparency)
        alpha = frame[:, :, 3] / 255.0 * opacity
        for c in range(3):  # Blend BGR channels
            image[y : y + h, x : x + w, c] = (
                alpha * frame[:, :, c] + (1 - alpha) * image[y : y + h, x : x + w, c]
            )


def change_frame_color(frame, color):
    """Change the color of the frame using blending."""
    if frame.shape[2] == 4:  # RGBA frame
        # Extract RGB channels
        frame_rgb = frame[:, :, :3]
        color_tint = np.full_like(frame_rgb, color)

        # Blend the tint color with the frame
        blended_frame = cv2.addWeighted(frame_rgb, 0.7, color_tint, 0.3, 0)

        # Reapply the alpha channel
        frame[:, :, :3] = blended_frame
    return frame


def create_thumbnail_strip(frames, current_frame_idx, strip_height, frame_width):
    """Create a strip of thumbnails."""
    num_thumbnails = len(frames)
    thumbnail_strip = np.zeros(
        (strip_height, num_thumbnails * strip_height, 3), dtype=np.uint8
    )
    thumbnail_size = strip_height

    for i, thumb_frame in enumerate(frames):
        thumbnail = cv2.resize(
            thumb_frame, (thumbnail_size, thumbnail_size), interpolation=cv2.INTER_AREA
        )
        if thumbnail.shape[2] == 4:  # Convert RGBA to BGR
            thumbnail = cv2.cvtColor(thumbnail, cv2.COLOR_BGRA2BGR)

        x_start = i * thumbnail_size
        x_end = x_start + thumbnail_size

        # Highlight the selected frame
        if i == current_frame_idx:
            cv2.rectangle(
                thumbnail,
                (0, 0),
                (thumbnail_size - 1, thumbnail_size - 1),
                (0, 255, 0),
                3,
            )

        thumbnail_strip[:, x_start:x_end] = thumbnail

    # Center the strip if it doesn't fill the entire frame width
    if thumbnail_strip.shape[1] < frame_width:
        padding = (frame_width - thumbnail_strip.shape[1]) // 2
        thumbnail_strip = cv2.copyMakeBorder(
            thumbnail_strip, 1, 1, padding, padding, cv2.BORDER_CONSTANT
        )

    return thumbnail_strip


def webcam_mode():
    """Run the webcam try-on mode."""
    cap = cv2.VideoCapture(0)
    current_frame_idx = 0
    scale_factor = 1.0
    rotation_angle = 0
    stem_x_adjust = 0  # Horizontal adjustment (left/right)
    stem_y_adjust = 0  # Vertical adjustment (up/down)
    opacity = 1.0  # Initial opacity
    snapshot_count = 0
    thumbnail_height = 100

    # Define colors for frame adjustments (BGR format)
    colors = {
        "red": (0, 0, 255),
        "green": (0, 255, 0),
        "blue": (255, 0, 0),
        "yellow": (0, 255, 255),
        "cyan": (255, 255, 0),
        "magenta": (255, 0, 255),
        "orange": (0, 165, 255),
        "purple": (128, 0, 128),
        "pink": (203, 192, 255),
        "default": (255, 255, 255),  # Default (no tint)
    }
    current_color = "default"

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to access the webcam.")
            break

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)
            landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])

            # Apply the current virtual frame with adjustments
            virtual_frame = frames[current_frame_idx].copy()
            virtual_frame = change_frame_color(virtual_frame, colors[current_color])
            virtual_frame, position = resize_and_rotate_frame(
                virtual_frame, landmarks, scale_factor, rotation_angle
            )

            # Fine-tune the position with manual adjustments
            adjusted_position = (
                position[0] + stem_x_adjust,
                position[1] + stem_y_adjust,
            )

            # Overlay the virtual frame onto the webcam feed
            overlay_frame(frame, virtual_frame, adjusted_position, opacity)

        # Add a thumbnail strip of the available frames
        thumbnail_strip = create_thumbnail_strip(
            frames, current_frame_idx, thumbnail_height, frame.shape[1]
        )
        frame = np.vstack((frame, thumbnail_strip))

        # Display the updated frame
        cv2.imshow("Virtual Frame Try-On", frame)

        # Handle keyboard inputs
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key to exit
            print("Exiting...")
            break
        elif key == ord("n"):  # Next frame
            current_frame_idx = (current_frame_idx + 1) % len(frames)
        elif key == ord("h"):  # Save a snapshot
            snapshot_filename = f"snapshot_{snapshot_count}.png"
            cv2.imwrite(snapshot_filename, frame[:-thumbnail_height])  # Exclude thumbnails
            snapshot_count += 1
            print(f"Snapshot saved as {snapshot_filename}")
        elif key == ord("="):  # Increase scale
            scale_factor += 0.1
        elif key == ord("-"):  # Decrease scale
            scale_factor = max(0.1, scale_factor - 0.1)
        elif key == ord("o"):  # Decrease opacity
            opacity = max(0.1, opacity - 0.1)
            print(f"Opacity decreased to {opacity:.1f}")
        elif key == ord("p"):  # Increase opacity
            opacity = min(1.0, opacity + 0.1)
            print(f"Opacity increased to {opacity:.1f}")
        elif key == ord("w"):  # Move frame up
            stem_y_adjust -= 5
        elif key == ord("s"):  # Move frame down
            stem_y_adjust += 5
        elif key == ord("a"):  # Move frame left
            stem_x_adjust -= 5
        elif key == ord("d"):  # Move frame right
            stem_x_adjust += 5
        elif key == ord("1"):  # Apply red tint
            current_color = "red"
            print("Changed frame color to Red.")
        elif key == ord("2"):  # Apply green tint
            current_color = "green"
            print("Changed frame color to Green.")
        elif key == ord("3"):  # Apply blue tint
            current_color = "blue"
            print("Changed frame color to Blue.")
        elif key == ord("4"):  # Apply yellow tint
            current_color = "yellow"
            print("Changed frame color to Yellow.")
        elif key == ord("5"):  # Apply cyan tint
            current_color = "cyan"
            print("Changed frame color to Cyan.")
        elif key == ord("6"):  # Apply magenta tint
            current_color = "magenta"
            print("Changed frame color to Magenta.")
        elif key == ord("7"):  # Apply orange tint
            current_color = "orange"
            print("Changed frame color to Orange.")
        elif key == ord("8"):  # Apply purple tint
            current_color = "purple"
            print("Changed frame color to Purple.")
        elif key == ord("9"):  # Apply pink tint
            current_color = "pink"
            print("Changed frame color to Pink.")
        elif key == ord("0"):  # Reset to default color
            current_color = "default"
            print("Reset frame color to Default.")

    # Release resources and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


def main():
    """Main menu."""
    print("Select an option:")
    print("1. Webcam Mode")
    choice = input("Enter your choice: ").strip()

    if choice == "1":
        webcam_mode()
    else:
        print("Invalid choice. Exiting.")


if __name__ == "__main__":
    main()
