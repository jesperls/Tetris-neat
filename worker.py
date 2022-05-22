import cv2
import neat
import cv2
import neat
from tetris import Tetris
import numpy as np
import math
import time

class Worker(object):
    def __init__(self, genome, config, namespace, game_queue, vision_queue):
        self.genome = genome
        self.config = config
        self.game_queue = game_queue
        self.vision_queue = vision_queue
        self.namespace = namespace
        self.fitness = 0
        self.tetris = Tetris()
        if self.namespace.visible_genome == -1:
            self.namespace.visible_genome = self.genome.key
        self.net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    def get_info(self, board):
        white_spaces = 0
        blocks = 0
        for i in range(len(board[0])):
            hit = False
            for j in range(len(board)):
                if board[j][i] != (255, 255, 255):
                    hit = True
                    if j > 2*len(board)/3:
                        blocks += 1
                elif j > 2*len(board)/3:
                    white_spaces += 1
        return blocks, white_spaces
    
    def get_inputs(self):
        inputs = []
        for i in self.tetris.board:
            row = []
            for j in i:
                if j != (255, 255, 255):
                    row.append(1)
                else:
                    row.append(0)
            inputs.extend(row)
        for x, y in self.tetris.current_piece.get_positions():
            inputs.append(x)
            inputs.append(y)
        return inputs

    def visible_managers(self):
        if self.fitness > self.namespace.max_fitness:
            self.namespace.max_fitness = self.fitness
            self.namespace.visible_genome = self.genome.key

        if self.namespace.visible_genome == -1:
            self.namespace.visible_genome = self.genome.key

        if self.namespace.visible_genome == self.genome.key and self.tetris.ticks%120 == 0:
            img = self.tetris.render()
            self.game_queue.put(img)
            cv2.waitKey(1)

    def run(self):
        self.tetris.reset()
        while not self.tetris.game_over:
            self.visible_managers()

            blocks, whitespaces = self.get_info(self.tetris.board)

            actions = self.net.activate(self.get_inputs())
            if actions[-1] > 0:
                actions[0], actions[1] = actions[1], actions[0]
                actions[-1] = 0
            self.tetris.actions = actions
            self.tetris.step()

            # if self.namespace.visible_genome == self.genome.key and self.tetris.ticks%60 == 0:
            #     print(blocks - whitespaces)
            self.fitness = self.tetris.score + blocks - whitespaces
        if self.namespace.visible_genome == self.genome.key:
            self.namespace.visible_genome = -1

        return self.fitness
