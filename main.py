import cv2
import numpy as np

# Part 1
def takeapic(pic_name):
    """Function takes picture from webcam and converts it
    into grayscale. Then saves image as jpg file.

    Parameters:
    pic_name: string
        The name of the picture to be taken.

    """
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow(pic_name, frame)
        # Press key "q" to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite(f'{pic_name}.jpg', frame)
            break
    cleanup(cap)


def show_pic(pic_name):
    """Function displays picture with colored line and rectangle
    drawn on top of it.

    Parameters:
    pic_name: string
        The name of the picture to be taken.
    """
    frame = cv2.imread(f'{pic_name}.jpg')
    frame = cv2.line(frame, (0, 0), (500, 500), (255, 0, 0), 20)
    frame = cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 5)
    cv2.imshow(pic_name, frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Part 2
def record_video(video_name):
    """Function records grayscale video and saves it in .avi
    file format.

    Parameters:
    video_name: string
        The name of the video to be taken.
    """
    cap = cv2.VideoCapture(0)
    # XVID works for some of the distributions of Linux.
    # Please, note that for your OS this option may not work.
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vwriter = cv2.VideoWriter(f'{video_name}.avi', fourcc, 20,
                                (640, 480), isColor=False)
    # The isColor option is required for saving video as grayscale.
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            vwriter.write(frame)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cleanup(cap)
    vwriter.release()


def display_video(video_name):
    """Function displays video and draws a line and a rectangle
    on top of it.

    Parameters:
    video_name: string
        The name of the video to be taken.
    """

    cap = cv2.VideoCapture(f'{video_name}.avi')
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.line(frame, (0, 0), (500, 500), (10, 255, 10), 20)
            frame = cv2.rectangle(frame, (100, 100), (400, 400), (255, 0, 0), 5)
            cv2.imshow(video_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cleanup(cap)


def cleanup(cap):
    """Utility function for cleanup after work with webcam.

    Parameters:
        cap: cv2.VideoCapture
        OpenCV object used to show and record videos.
    """
    cap.release()
    cv2.destroyAllWindows()


def main():
    # takeapic('lol')
    # show_pic('lol')
    # record_video('lol')
    display_video('lol')


if __name__ == "__main__":
    main()