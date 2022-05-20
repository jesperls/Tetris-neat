import cv2
import neat
import cv2
import neat


class Worker(object):
    def __init__(self, genome, config, namespace, game_queue, vision_queue):
        self.genome = genome
        self.config = config
        self.game_queue = game_queue
        self.vision_queue = vision_queue
        self.namespace = namespace
        self.net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    def run(self):
        pass