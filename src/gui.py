from network import *
import pygame
from pygame.locals import *
import sys

pygame.init()

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
FPS = 60
fps_clock = pygame.time.Clock()

SCALE = 20
WIDTH = 28 * SCALE
HEIGHT = 28 * SCALE

WINDOW = pygame.display.set_mode((WIDTH, HEIGHT))

cells = np.zeros((28, 28))

def fill_nearby(mouse_x, mouse_y):
    y = int((mouse_y)/SCALE)
    x = int(mouse_x/SCALE)
    cells[y][x] = 1

    if x <= 1 or x >= 27:
        return
    if y <= 1 or y >= 27:
        return

    cells[int((mouse_y)/SCALE) + 1][int(mouse_x/SCALE)] = 1
    cells[int((mouse_y)/SCALE) - 1][int(mouse_x/SCALE)] = 1

    cells[int((mouse_y)/SCALE)][int(mouse_x/SCALE) + 1] = 1
    cells[int((mouse_y)/SCALE)][int(mouse_x/SCALE) - 1] = 1

def game_loop():
    print("start game loop!")
    dragging = False

    name = "100hidden"
    W1, B1, W2, B2 = np.load(f'models/{name}/master.npy', allow_pickle=True)
    while True:
        WINDOW.fill((255,0,0))

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                dragging = True
                mx, my = event.pos
                fill_nearby(mx, my)        
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                dragging = False
                mx, my = event.pos
                fill_nearby(mx, my)
            elif event.type == pygame.MOUSEMOTION:
                if dragging:
                    mx, my = event.pos
                    fill_nearby(mx, my)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:
                    # cells = np.zeros((28, 28)) something is buggy about this line of code
                    print("clear!")
                if event.key == pygame.K_RETURN:
                    print("predict!")
                    inputs = cells.flatten().reshape((784, 1))

                    count = 0
                    for pixel in cells.flatten():
                        if pixel != 0:
                            print("⬜️", end='')
                        else:
                            print("⬛️", end='')
                            
                        count += 1
                        if count % 28 == 0: print("\n", end='') # add a new line

                    prediction = make_predictions(inputs, W1, B1, W2, B2)
                    print(f"Prediction: {prediction[0]}")
                    for y, row in enumerate(cells):
                        for x, val in enumerate(row):
                            cells[y][x] = 0
        
        for y, row in enumerate(cells):
            for x, val in enumerate(row):
                col = BLACK
                if val == 1: 
                    col = WHITE
                pygame.draw.rect(WINDOW, col, (x * SCALE, y * SCALE, SCALE, SCALE))      

        pygame.display.update()
        fps_clock.tick(FPS)

if __name__ == '__main__':   
    name = "100hidden"
    # W1, B1, W2, B2 = np.load(f'models/{name}/master.npy', allow_pickle=True)
    # W1, B1, W2, B2 = gradient_descent(X_train, Y_train, learning_rate=0.4, iterations=600, n_hidden=100)
    # dev_predictions = make_predictions(X_dev, W1, B1, W2, B2)
    # print(f"Test Accuracy: {get_accuracy(dev_predictions, Y_dev)}")
    # download_model(W1, B1, W2, B2, name)
    # game_loop()

    raw_data = np.load('data/mod_train.npy')
    print(raw_data.shape)
    raw_data = np.array(pd.read_csv('data/train.csv'))
    print(raw_data.shape)


    # for y, row in enumerate(data):
    #     for x, val in enumerate(row):
    #         if x == 0: continue
    #         if val != 0:
    #             data[y][x] = 1

    # NEED TO MAKE LABELS FOR THE CSV BEFORE SAVING IT
    # pd.DataFrame(data).to_csv('data/mod_train.csv')
    # or you could save the modified data as a .npy file for easy access
    # np.save(f"data/mod_train.npy", data)
    
    count = 0
    for pixel in data[0][1:]:
        if pixel == 1:
            print("⬜️", end='')
        else:
            print("⬛️", end='')
            
        count += 1
        if count % 28 == 0: print("\n", end='') # add a new line