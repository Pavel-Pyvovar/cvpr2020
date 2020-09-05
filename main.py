import cv2

def takeapic(pic_name):
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow(pic_name, gray)
        pic = np.copy(gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            pic = cv2.line(pic, (0, 0), (500, 500), (255, 0, 0), 20)
            pic = cv2.rectangle(pic, (100, 100), (400, 400), (255, 0, 0), 5)
            cv2.imwrite(f'{pic_name}.jpg', pic)
            break
    cleanup(cap)


def show_pic(pic_name):
    pic = cv2.imread(pic_name)
    cv2.imshow(pic_name, pic)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def record_video(video_name):
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vwriter = cv2.VideoWriter(f'{video_name}.avi', fourcc, 20, (640, 480))

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.line(frame, (0, 0), (500, 500), (255, 0, 0), 20)
            frame = cv2.rectangle(frame, (100, 100), (400, 400), (255, 0, 0), 5)
            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cleanup(cap)
    vwriter.release()


def display_video(video_name):
    cap = cv2.VideoCapture(f'{video_name}.avi')
    while cap.isOpened():
        ret, frame = cap.read()
        print(ret)
        cv2.imshow(video_name, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cleanup(cap)


def cleanup(cap):
    cap.release()
    cv2.destroyAllWindows()


def main():
    # record_video('kek')
    display_video('kek')


if __name__ == "__main__":
    main()