from lib.controller import Controller
import cv2

char_list = ['0', '2', '6', 'B', 'H', 'I', 'P']

if __name__ == '__main__':
    controller = Controller(gpu_id=0)

    for char in char_list:
        img = cv2.imread(f'assets/data/{char}.jpg')
        char_pred = controller.infer([img])[0]

        print("gt: {}\tpred: {}".format(char, char_pred))
