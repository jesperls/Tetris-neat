import neat
import multiprocessing
import os
from worker import Worker
import threading
import cv2

def eval_genome(genome, config, namespace, game_queue, vision_queue):
    worker = Worker(genome, config, namespace, game_queue, vision_queue)
    fitness = worker.run()
    return fitness

def draw(game_queue):
    while True:
        if not game_queue.empty():
            cv2.imshow("Tetris, but smart", game_queue.get())
            cv2.waitKey(1)

def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_file)

    manager = multiprocessing.Manager()
    game_queue = manager.Queue()
    vision_queue = manager.Queue()
    namespace = manager.Namespace()

    namespace.visible_genome = -1
    namespace.max_fitness = 0
    namespace.running_best = False
    
    draw_manager = threading.Thread(target=draw, args=(game_queue,))
    draw_manager.start()

    while True:
        p = neat.Population(config)

        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)

        pe = neat.ParallelEvaluator(16, eval_genome)

        winner = p.run(pe.evaluate, namespace, game_queue, vision_queue)

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    run(config_path)