import pygame
import random
import math
import os
import numpy as np

# Game Variables
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 100, 0)

display_width = 1400
display_height = 800
FPS = 60
pipe_gap = 200
between_pipe = 200
pipe_width = 100
pipe_speed = 6
score = 0
velocity = 10
pipe_count = display_width // (pipe_width + between_pipe) + 2

# Genetic Variables
population = 350
hidden_nodes = 8
inp = 4
bias1 = np.random.uniform()
bias2 = np.random.uniform()

game_folder = os.path.dirname(__file__)
master_parameters = [np.zeros(shape=(inp, hidden_nodes)), np.zeros(shape=hidden_nodes + 1)]

# Genetic Algorithm Functions
def sigmoid(value):
    value = float(math.exp(-value))
    value = float(value + 1)
    value = float(1 / value)
    return value

def nn(arr, paras, bias2):
    hidden_activations = np.dot(arr, paras[0])
    hidden_activations = [bias2] + list(map(sigmoid, hidden_activations))
    return sigmoid(np.dot(hidden_activations, paras[1]))

def mutate(master):
    return [param + np.random.normal(scale=1, size=param.shape) for param in master]

def make_parameters(master, population):
    para_list = [master]
    for _ in range(population - 1):
        para_list.append(mutate(master))
    return para_list

# Pygame Sprite Classes
class Bird(pygame.sprite.Sprite):
    def __init__(self, x_loc, y_loc, velocity):
        super(Bird, self).__init__()
        self.check = 0
        self.velocity = velocity
        self.x_loc = x_loc
        self.y_loc = y_loc
        self.image = pygame.image.load(os.path.join(game_folder, "index.png")).convert()
        self.image.set_colorkey(WHITE)
        self.image = pygame.transform.scale(self.image, (40, 40))
        self.rect = self.image.get_rect()
        self.rect.center = (x_loc, y_loc)
        self.mask = pygame.mask.from_surface(self.image)

    def update(self):
        self.rect.y += self.velocity
        self.velocity += 1

    def jump(self):
        self.velocity = -10

    def boundary_collison(self):
        if self.rect.bottom + 100 >= display_height or self.rect.top <= 0:
            return True

    def bird_center(self):
        return self.rect.center

    def vel(self):
        return self.velocity

class UpperPipe(pygame.sprite.Sprite):
    def __init__(self, pipe_x, pipe_height, pipe_speed):
        super(UpperPipe, self).__init__()
        self.pipe_speed = pipe_speed
        self.pipe_height = pipe_height
        self.image = pygame.Surface((pipe_width, pipe_height))
        self.image.fill(GREEN)
        self.image.set_colorkey(WHITE)
        self.rect = self.image.get_rect()
        self.rect.x = pipe_x
        self.rect.y = 0
        self.mask = pygame.mask.from_surface(self.image)

    def update(self):
        self.rect.x -= self.pipe_speed

    def x_cord(self):
        return self.rect.x

    def y_cord(self):
        return self.rect.y + self.pipe_height

class LowerPipe(pygame.sprite.Sprite):
    def __init__(self, pipe_x, pipe_height, pipe_speed):
        super(LowerPipe, self).__init__()
        self.pipe_speed = pipe_speed
        self.image = pygame.Surface((pipe_width, display_height - (pipe_gap + pipe_height)))
        self.image.fill(GREEN)
        self.image.set_colorkey(WHITE)
        self.rect = self.image.get_rect()
        self.rect.x = pipe_x
        self.rect.y = pipe_height + pipe_gap
        self.mask = pygame.mask.from_surface(self.image)

    def update(self):
        self.rect.x -= self.pipe_speed

    def x_cord(self):
        return self.rect.x

    def y_cord(self):
        return self.rect.y

# Will create a new set of upper and lower pipe every time an existing pipe moves off the screen.
def new_pipe(pipe):
    pipe_x = pipe[0].x_cord() + between_pipe + pipe_width
    pipe_height = (round(np.random.uniform(0.15, 0.85), 2)) * (display_height - pipe_gap)
    upper = UpperPipe(pipe_x, pipe_height, pipe_speed)
    lower = LowerPipe(pipe_x, pipe_height, pipe_speed)
    add_pipe = [upper, lower]
    pipe_group.add(upper)
    pipe_group.add(lower)
    return add_pipe

def init_para():
    parameter_list = []
    for _ in range(population):
        m_parameters = [np.random.normal(size=(inp, hidden_nodes)), np.random.normal(size=hidden_nodes + 1)]
        parameter_list.append(m_parameters)
    return parameter_list

def init_bias():
    return np.random.normal(size=(population, 2))

# Main Game Run Function
def run_game(generation, score, bias_list):
    myfont = pygame.font.SysFont("monospace", 16)
    run_score = 0
    best_score = []
    best_bias = []
    manage = []
    global check
    global gameExit
    global master_parameters
    global pipe_collision
    global pipe
    cur_index = 0
    while not gameExit:
        clock.tick(FPS)
        gameDisplay.fill(WHITE)
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    gameExit = True
                if event.key == pygame.K_SPACE:
                    for bird_index in range(len(bird_list)):
                        if check[bird_index] == 0:
                            bird = bird_list[bird_index]
                            bird.jump()
                            if bird.boundary_collison():
                                pygame.sprite.Sprite.kill(bird)
                                check[bird_index] = 1
                                return parameter_list[bird_index], run_score, bias_list[bird_index]
                            for x in pipe_collision:
                                c = 0
                                if pygame.sprite.collide_rect(bird, x):
                                    bird_hits = pygame.sprite.spritecollide(bird, pipe_group, False, pygame.sprite.collide_mask)
                                    if bird_hits:
                                        c = 1
                                        pygame.sprite.Sprite.kill(bird)
                                        check[bird_index] = 1
                                        return parameter_list[bird_index], run_score, bias_list[bird_index]

        for bird_index in range(len(bird_list)):
            if check[bird_index] == 0:
                bias1, bias2 = bias_list[bird_index][0], bias_list[bird_index][1]
                bird = bird_list[bird_index]
                arr = [bias1]
                bird_x, bird_y = bird.bird_center()
                arr.append(bird.vel())
                xpip = pipe_collision[0].x_cord()
                draw_x, draw_y = 0, 0

                if bird_x <= xpip:
                    xpip = pipe_collision[0].x_cord() / display_width
                    ypip = (pipe_collision[0].y_cord() + (pipe_gap // 2)) / display_height
                    arr.append((bird_x / display_width) - xpip)
                    arr.append((bird_y / display_height) - ypip)
                    draw_x = pipe_collision[0].x_cord()
                    draw_y = pipe_collision[0].y_cord() + (pipe_gap // 2)
                elif bird_x <= (xpip + (pipe_width // 2)):
                    xpip = (pipe_collision[0].x_cord() + (pipe_width // 2)) / display_width
                    ypip = (pipe_collision[0].y_cord() + (pipe_gap // 2)) / display_height
                    arr.append((bird_x / display_width) - xpip)
                    arr.append((bird_y / display_height) - ypip)
                    draw_x = pipe_collision[0].x_cord() + (pipe_width // 2)
                    draw_y = pipe_collision[0].y_cord() + (pipe_gap // 2)
                else:
                    xpip = (pipe_collision[0].x_cord() + pipe_width) / display_width
                    ypip = (pipe_collision[0].y_cord() + (pipe_gap // 2)) / display_height
                    arr.append((bird_x / display_width) - xpip)
                    arr.append((bird_y / display_height) - ypip)
                    draw_x = pipe_collision[0].x_cord() + pipe_width
                    draw_y = pipe_collision[0].y_cord() + (pipe_gap // 2)

                pygame.draw.circle(gameDisplay, RED, (draw_x, int(draw_y)), 5)
                direct_distance = (arr[1] ** 2) + (arr[2] ** 2)
                direct_distance = math.sqrt(direct_distance)
                fitness = run_score - direct_distance
                out = nn(arr, parameter_list[bird_index], bias2)
                if out > 0.5:
                    bird.jump()

                # Check for collisions
                if bird.boundary_collison():
                    pygame.sprite.Sprite.kill(bird)
                    best_score.append(fitness)
                    best_bias.append([bias1, bias2])
                    check[bird_index] = 1

                for x in pipe_collision:
                    c = 0
                    if pygame.sprite.collide_rect(bird, x) and check[bird_index] == 0:
                        bird_hits = pygame.sprite.spritecollide(bird, pipe_group, False, pygame.sprite.collide_mask)
                        if bird_hits:
                            c = 1
                            pygame.sprite.Sprite.kill(bird)
                            best_score.append(fitness)
                            best_bias.append([bias1, bias2])
                            check[bird_index] = 1
                            break
                    if c == 1:
                        break

            if sum(check) == len(check):
                bias_return = []
                if max(best_score) > score:
                    try:
                        master_parameters = parameter_list[list(best_score).index(max(best_score))]
                        score = max(best_score)
                        bias_return = bias_list[list(best_bias).index(max(best_bias))]
                    except:
                        print("Debug Check 1", sum(check))
                        print("Debug Check 1", len(best_score))
                        pygame.quit()
                        quit()
                return master_parameters, score, bias_return

        sprites.update()
        pipe_group.update()
        sprites.draw(gameDisplay)
        pipe_group.draw(gameDisplay)

        if (pipe[0].x_cord()) + pipe_width <= 0:
            for k in pipe:
                pygame.sprite.Sprite.kill(k)
            pipe = pipe_list[1]
            del pipe_list[0]
            pipe_list.append(new_pipe(pipe_list[-1]))

        if (pipe_collision[0].x_cord() + pipe_width) < x_loc:
            pipe_collision = pipe_list[1]

        gen = myfont.render("Generation {0}".format(generation), 1, (0, 0, 0))
        highest = myfont.render("Highest Score {0}".format(int(round(score))), 1, (0, 0, 0))
        current = myfont.render("Current Score {0}".format(run_score), 1, (0, 0, 0))
        gameDisplay.blit(gen, (5, 10))
        gameDisplay.blit(highest, (5, 35))
        gameDisplay.blit(current, (5, 60))
        run_score += 1
        pygame.display.flip()

score = 0
generation = 1
parameter_list = init_para()
bias_list = init_bias()
threshold_score = 0
threshold = 5
threshold_count = 0
bias = []

while True:
    pygame.init()
    gameDisplay = pygame.display.set_mode((display_width, display_height))
    clock = pygame.time.Clock()
    pygame.display.set_caption("Drone Simulation - NeuroEvolution")

    x_loc = display_width // 8
    y_loc = display_height // 2

    sprites = pygame.sprite.Group()
    pipe_group = pygame.sprite.Group()
    y_locations = np.random.randint(low=0, high=display_height, size=population)
    bird_list = []
    for make_bird in range(population):
        bird = Bird(x_loc, y_locations[make_bird], velocity)
        bird_list.append(bird)
        sprites.add(bird)

    pipe_list = []
    init_pipe_x = 500
    for make in range(pipe_count):
        pipe_x = init_pipe_x + ((between_pipe + pipe_width) * make)
        pipe_height = (round(np.random.uniform(0.15, 0.85), 2)) * (display_height - pipe_gap)
        upper = UpperPipe(pipe_x, pipe_height, pipe_speed)
        lower = LowerPipe(pipe_x, pipe_height, pipe_speed)
        add_pipe = [upper, lower]
        pipe_list.append(add_pipe)
        pipe_group.add(upper)
        pipe_group.add(lower)
    pipe = pipe_list[0]
    pipe_collision = pipe_list[0]
    gameExit = False
    check = [0] * population
    cpy_mp, cpy_score, cpy_bias = master_parameters, score, bias
    master_parameters, score, bias = run_game(generation, score, bias_list)
    if len(bias) == 0:
        master_parameters, score, bias = cpy_mp, cpy_score, cpy_bias
    pygame.quit()
    print("Generation =", generation, "---------- Highest Score =", int(score))

    if threshold_score < score:
        threshold_score = score
        threshold_count = 0
    else:
        threshold_count += 1
    if threshold_count == threshold:
        threshold_count = 0
        print("********** Purged **********")
        parameter_list = init_para()
        bias_list = init_bias()
    else:
        parameter_list1 = [master_parameters] * int(population * 0.4)
        parameter_list2 = make_parameters(master_parameters, int(population * 0.6))
        parameter_list = parameter_list1 + parameter_list2
        bias_list1 = [bias] * int(population * 0.4)
        bias_list2 = make_parameters(bias, int(population * 0.6))
        bias_list = bias_list1 + bias_list2

    generation += 1
