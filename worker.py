import cv2
import neat
import cv2
import neat
from tetris import Tetris
import numpy as np

class Worker(object):
    def __init__(self, genome, config, namespace, game_queue, vision_queue):
        self.genome = genome
        self.config = config
        self.game_queue = game_queue
        self.vision_queue = vision_queue
        self.namespace = namespace
        self.tetris = Tetris()
        if self.namespace.visible_genome == -1:
            self.namespace.visible_genome = self.genome.key
        self.net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    def run(self):
        self.tetris.reset()
        while not self.tetris.game_over:
            if self.namespace.visible_genome == self.genome.key:
                img = self.tetris.render()
                if (type(img) == np.ndarray):
                    self.game_queue.put(img)
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
            if len(inputs) != 228:
                for i in range(228-len(inputs)):
                    inputs.append(0)
            inputs.append(self.tetris.current_piece.x)
            inputs.append(self.tetris.current_piece.y)

            actions = self.net.activate(inputs)

            self.tetris.actions = actions
            self.tetris.step()
        if self.namespace.visible_genome == self.genome.key:
            self.namespace.visible_genome = -1
        return self.tetris.score
