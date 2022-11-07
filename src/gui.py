from network import *
import pygame
from pygame.locals import *
import sys
import matplotlib.pyplot as plt

pygame.init()
fps_clock = pygame.time.Clock()

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
FPS = 60

SCALE = 20
WIDTH = 28 * SCALE
HEIGHT = 28 * SCALE

WINDOW = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("MNIST Digit Classifier")

font = pygame.font.SysFont('timesnewroman',  20)

cells = np.zeros((28, 28))

def fill_nearby(mouse_x, mouse_y):
    y = int((mouse_y)/SCALE)
    x = int(mouse_x/SCALE)
    
    if x < 4 or x > 22: return
    if y < 4 or y > 22: return
    
    cells[y][x] = 1

    if x >= 27 or y >= 27:
        return

    cells[int((mouse_y)/SCALE) + 1][int(mouse_x/SCALE)] = 1
    cells[int((mouse_y)/SCALE)+ 1][int(mouse_x/SCALE) + 1] = 1
    cells[int((mouse_y)/SCALE)][int(mouse_x/SCALE) + 1] = 1

def game_loop():
    print("start game loop!")
    dragging = False
    prediction = -1

    name = "mod_100hidden"
    W1, B1, W2, B2 = np.load(f'models/{name}/master.npy', allow_pickle=True)
    while True:
        WINDOW.fill((255,0,0))

        for y, row in enumerate(cells):
            for x, val in enumerate(row):
                col = BLACK
                if (x < 4 or x > 23) or (y < 4 or y > 23):
                    col = (240, 240, 240) 
                if val == 1: 
                    col = WHITE
                pygame.draw.rect(WINDOW, col, (x * SCALE, y * SCALE, SCALE, SCALE), width=0)      

        prediction_label = font.render(f" Prediction: { prediction } ", True, BLACK, (WHITE)) 
        lbl_rect = prediction_label.get_rect()
        lbl_rect.center = (28 * SCALE / 2, 30)
        WINDOW.blit(prediction_label, lbl_rect)

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
                    for y, row in enumerate(cells):
                        for x, val in enumerate(row):
                            cells[y][x] = 0
                if event.key == pygame.K_RETURN:
                    inputs = cells.flatten().reshape((784, 1))

                    count = 0
                    for pixel in cells.flatten():
                        if pixel != 0:
                            print("⬜️", end='')
                        else:
                            print("⬛️", end='')
                            
                        count += 1
                        if count % 28 == 0: print("\n", end='') # add a new line

                    prediction = make_predictions(inputs, W1, B1, W2, B2)[0]
                    print(f"Prediction: {prediction}")
                    
        pygame.display.update()
        fps_clock.tick(FPS)

def image_of_weights(W):
    shape = W.shape
    i = 0
    img = W1[0].reshape((28, 28))
    print(img)
    while True:
        pygame.display.set_caption(f"Weight {i + 1}")
        for y, row in enumerate(img):
            for x, pixel in enumerate(row):
                col = BLACK
                if pixel < 0:
                    col = (abs(pixel) * 255, 0, 0, abs(pixel) * 255)
                elif pixel > 0:
                    col = (0, 0, abs(pixel) * 255, abs(pixel) * 255)
                pygame.draw.rect(WINDOW, col, (y * SCALE, x * SCALE, SCALE, SCALE), width=0)

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    if ((i + 1) <= (shape[0] - 1)): i += 1
                    else: i = 0
                elif event.key == pygame.K_LEFT:
                    if ((i - 1) >= (0)): i -= 1
                    else: i = 15
                    # move to the next image of weights
                img = W1[i].reshape((28, 28))
        pygame.display.update()
        fps_clock.tick(FPS)

if __name__ == '__main__': 
    np.set_printoptions(suppress=True)

    name = "100hidden"
    W1, B1, W2, B2 = np.load(f'models/{name}/master.npy', allow_pickle=True)
    dev_predictions = make_predictions(X_dev, W1, B1, W2, B2)
    print(f"Test Accuracy: {get_accuracy(dev_predictions, Y_dev)}")

    # W1, B1, W2, B2 = gradient_descent(X_train, Y_train, learning_rate=0.4, iterations=600, n_hidden=100)
    # download_model(W1, B1, W2, B2, name)

    img = reverse_engineer_image().x # .x grabs the succesful array
    # img = img.reshape((28, 28))
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')

    # define location of the plot
    x3 = np.zeros((28, 28))
    num = 0
    for i in range(28):
        for j in range(28):
            x3[i][j] = num
        num = num + 1

    x3 = x3.flatten()

    y3 = np.zeros((28, 28))
    for i in range(28):
        y3[i] = np.arange(28)
    y3 = y3.flatten()
    
    z3 = np.zeros(784)
    # depth/width/height of the bar graphs
    dx = np.ones(784)
    dy = np.ones(784)
    dz = img
    
    ax1.bar3d(x3, y3, z3, dx, dy, dz)


    ax1.set_xlabel('x axis')
    ax1.set_ylabel('y axis')
    ax1.set_zlabel('z axis')

    plt.show()

    # _,_,_,A2 = feed_forward(img, W1, B1, W2, B2)
    # out = list(flatten_output(A2))
    # prediction = out.index(max(out))
    # print(f"Prediction: { prediction }") 

    # image_of_weights(W1)