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

# def input_scanner():
#     while True:
#         if keyboard.is_pressed('b'):
#             run_best()
#             while keyboard.is_pressed('b'):
#                 pass

def draw(game_queue, vision_queue):
    while True:
        if not game_queue.empty():
            pass
            # cv2.imshow("MARI/O", game_queue.get())
        if not vision_queue.empty():
            pass
            # cv2.imshow("MARI/O Vision", vision_queue.get())
        cv2.waitKey(1)

def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_file)

    # Create the population, which is the top-level object for a NEAT run.

    # Add a stdout reporter to show progress in the terminal.
    #p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-108')

    # p.add_reporter(neat.StdOutReporter(True))
    # stats = neat.StatisticsReporter()
    # p.add_reporter(stats)
    # p.add_reporter(neat.Checkpointer(10))

    # IFFY DonutPlains5 Forest1 YoshiIsland2 (5000)
    # Tested YoshiIsland2 DonutPlains1 YoshiIsland1

    manager = multiprocessing.Manager()


    game_queue = manager.Queue()
    vision_queue = manager.Queue()
    namespace = manager.Namespace()
    
    draw_manager = threading.Thread(target=draw, args=(game_queue, vision_queue))
    draw_manager.start()

    while True:
        p = neat.Population(config)

        pe = neat.ParallelEvaluator(int(multiprocessing.cpu_count()/2), eval_genome)

        winner = p.run(pe.evaluate, namespace, game_queue, vision_queue)

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward-mario')
    run(config_path)