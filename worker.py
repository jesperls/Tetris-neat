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
    
    """takes in the tetris board and calculates the average height differance between columns"""
    def get_height_diff(self, board):
        heights = []
        for i in range(len(board[0])):
            for j in range(len(board)):
                if board[j][i] != (255, 255, 255):
                    heights.append(j-1)
                    break
                if j == len(board) - 1:
                    heights.append(j)
        return np.std(heights), np.mean(heights)

    def run(self):
        self.tetris.reset()
        while not self.tetris.game_over:
            if self.namespace.visible_genome == self.genome.key:
                img = self.tetris.render()
                if (type(img) == np.ndarray):
                    self.game_queue.put(img)
                    cv2.waitKey(1)

            if self.namespace.visible_genome == -1 and not self.namespace.running_best:
                self.namespace.visible_genome = self.genome.key
            
            inputs = []
            for i in self.tetris.board:
                for j in i:
                    if j != (255, 255, 255):
                        inputs.append(1)
                    else:
                        inputs.append(0)

            for x, y in self.tetris.current_piece.get_positions():
                inputs.append(x)
                inputs.append(y)
            
            height_diffs = self.get_height_diff(self.tetris.board)
            inputs.extend([height_diffs[0], height_diffs[1]])

            actions = self.net.activate(inputs)

            self.tetris.actions = actions
            self.tetris.step()

            self.fitness =  self.tetris.score + int(self.tetris.ticks/300)

        if self.namespace.visible_genome == self.genome.key:
            self.namespace.visible_genome = -1

        return self.fitness
