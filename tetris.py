import time
import random
import numpy as np
import cv2
import keyboard
import fpstimer

tile_size = 32

tetrominos = [{"blocks": [[0, 1], [1, 1], [2, 1], [3, 1]], "color":(255, 255, 0), "center": (1.5,1.5), "id":1}, 
              {"blocks": [[0, 1], [1, 1], [1, 0], [2, 0]], "color":(0, 255, 0), "center": (1,0), "id":2},
              {"blocks": [[0, 0], [0, 1], [1, 1], [2, 1]], "color":(255, 0, 0), "center":(1,1), "id":3},
              {"blocks": [[0, 1], [1, 1], [2, 1], [2, 0]], "color":(0, 127, 255), "center":(1,1), "id":4},
              {"blocks": [[0, 1], [1, 1], [1, 0], [2, 1]], "color":(128, 0, 128), "center":(1,1), "id":5},
              {"blocks": [[0, 0], [1, 0], [1, 1], [2, 1]], "color":(0, 0, 255), "center":(1,0), "id":6},
              {"blocks": [[0, 0], [1, 0], [1, 1], [0, 1]], "color":(0, 255, 255), "center":(0,0), "id":7}]

upscaling = 16

levels = [{"gravity" : 48, "to_next":10}, {"gravity" : 43, "to_next":20}, {"gravity" : 38, "to_next":30}, 
          {"gravity" : 33, "to_next":40}, {"gravity" : 38, "to_next":50}, {"gravity" : 23, "to_next":60},
          {"gravity" : 18, "to_next":70}, {"gravity" : 13, "to_next":80}, {"gravity" : 8, "to_next":90}, 
          {"gravity" : 6, "to_next":100}, {"gravity" : 5, "to_next":100}, {"gravity" : 5, "to_next":100}, 
          {"gravity" : 5, "to_next":100}, {"gravity" : 4, "to_next":100}, {"gravity" : 4, "to_next":100}, 
          {"gravity" : 4, "to_next":100}, {"gravity" : 3, "to_next":110}, {"gravity" : 3, "to_next":120}, 
          {"gravity" : 3, "to_next":130}, {"gravity" : 2, "to_next":140}, {"gravity" : 2, "to_next":150}, 
          {"gravity" : 2, "to_next":160}, {"gravity" : 2, "to_next":170}, {"gravity" : 2, "to_next":180}, 
          {"gravity" : 2, "to_next":190}, {"gravity" : 2, "to_next":200}, {"gravity" : 2, "to_next":200}, 
          {"gravity" : 2, "to_next":200}, {"gravity" : 2, "to_next":200}, {"gravity" : 1, "to_next":99999999}]

dimensions = [10, 22]

class Tetris(object):
    def __init__(self):
        self.game_over = False
        self.score = 0
        self.lines = 0
        self.level = 0
        self.progress = 0
        self.ticks = 0
        self.actions = [0, 0, 0, 0, 0]
        self.pressed_keys = [0, 0, 0, 0, 0]
        self.board = [[(255, 255, 255)]*dimensions[0] for _ in range(dimensions[1])]
        self.current_piece = self.get_next_piece()
        self.next_piece = self.get_next_piece()
    
    def reset(self):
        self.game_over = False
        self.score = 0
        self.lines = 0
        self.level = 0
        self.progress = 0
        self.ticks = 0
        self.actions = [0, 0, 0, 0, 0]
        self.pressed_keys = [0, 0, 0, 0, 0]
        self.board = [[(255, 255, 255)]*dimensions[0] for _ in range(dimensions[1])]
        self.current_piece = self.get_next_piece()
        self.next_piece = self.get_next_piece()

    def start(self):
        self.reset()
        print(self.run())
    
    def handle_input(self):
        self.refresh_input()
        if (keyboard.is_pressed('d') or keyboard.is_pressed('right')) and self.pressed_keys[0] == 0:
            self.actions[0] = 1
            self.pressed_keys[0] = 1
        if (keyboard.is_pressed('a') or keyboard.is_pressed('left')) and self.pressed_keys[1] == 0:
            self.actions[1] = 1
            self.pressed_keys[1] = 1
        if (keyboard.is_pressed('s') or keyboard.is_pressed('down')) and self.pressed_keys[2] == 0:
            self.actions[2] = 1
            self.pressed_keys[2] = 1
        if (keyboard.is_pressed('w') or keyboard.is_pressed('up')) and self.pressed_keys[3] == 0:
            self.actions[3] = 1
            self.pressed_keys[3] = 1
        if keyboard.is_pressed('space') and self.pressed_keys[4] == 0:
            self.actions[4] = 1
            self.pressed_keys[4] = 1
        if sum(self.actions) > 0:
            return True
        return False

    def refresh_input(self):
        if not (keyboard.is_pressed('d') or keyboard.is_pressed('right')) and self.pressed_keys[0] == 1:
            self.pressed_keys[0] = 0
        if not (keyboard.is_pressed('a') or keyboard.is_pressed('left')) and self.pressed_keys[1] == 1:
            self.pressed_keys[1] = 0
        if not (keyboard.is_pressed('s') or keyboard.is_pressed('down')) and self.pressed_keys[2] == 1:
            self.pressed_keys[2] = 0
        if not (keyboard.is_pressed('w') or keyboard.is_pressed('up')) and self.pressed_keys[3] == 1:
            self.pressed_keys[3] = 0
        if not keyboard.is_pressed('space') and self.pressed_keys[4] == 1:
            self.pressed_keys[4] = 0

    def handle_actions(self):
        if self.actions[0] > 0:
            self.move_right()
            self.actions[0] = 0
        if self.actions[1] > 0:
            self.move_left()
            self.actions[1] = 0
        if self.actions[2] > 0:
            self.move_down()
            self.actions[2] = 0
        if self.actions[3] > 0:
            self.rotate()
            self.actions[3] = 0
        if self.actions[4] > 0:
            self.drop_down()
            self.actions[4] = 0

    def get_next_piece(self):
        selected = random.choice(tetrominos)
        # selected = tetrominos[4]
        tetromino = Tetromino(selected["blocks"], selected["color"], selected["center"], selected["id"])
        for x, y in tetromino.get_positions():
            if self.board[y][x] != (255, 255, 255):
                self.game_over = True
        return tetromino
    
    def move_check(self):
        for x, y in self.current_piece.get_positions():
            if x < 0 or x >= dimensions[0] or y >= dimensions[1] or self.board[y][x] != (255, 255, 255):
                return True
        return False

    def stop_check(self):
        blocks = self.current_piece.get_positions()
        for x1, y1 in blocks:
            if y1 >= dimensions[1]-1 or self.board[y1+1][x1] != (255, 255, 255):
                for x2, y2 in blocks:
                    self.board[y2][x2] = self.current_piece.color
                self.current_piece = self.next_piece
                self.next_piece = self.get_next_piece()
                return False
        return True

    def draw_board(self):
        img = np.zeros((dimensions[1], dimensions[0], 3), dtype=np.uint8)
        for y in range(0, dimensions[1]):
            for x in range(0, dimensions[0]):
                img[y, x] = self.board[y][x]
        return img

    def draw_grid(self, img):
        for i in range(0, dimensions[1]*upscaling):
            for j in range(0, dimensions[0]*upscaling):
                if i % upscaling == 0:
                    img[i, j] = (0, 0, 0)
                elif j % upscaling == 0:
                    img[i, j] = (0, 0, 0)

    def draw_piece(self, img):
        for x, y in self.current_piece.get_positions():
            img[y, x] = self.current_piece.color

    def find_line(self):
        lines = 0
        for y in range(0, 22):
            if all(self.board[y][x] != (255, 255, 255) for x in range(0, dimensions[0])):
                lines += 1
                self.board.pop(y)
                self.board.insert(0, [(255, 255, 255)]*dimensions[0])
                self.lines += 1
                if self.lines >= levels[self.level]["to_next"]:
                    self.level += 1
                    self.lines = 0
        if lines == 1:
            self.score += 40*(self.level+1)
        elif lines == 2:
            self.score += 100*(self.level+1)
        elif lines == 3:
            self.score += 300*(self.level+1)
        elif lines == 4:
            self.score += 1200*(self.level+1)

    def move_piece(self, dx, dy):
        self.current_piece.move(dx, dy)
        if self.move_check():
            self.current_piece.move(-dx, -dy)
            return False
        return True

    def move_down(self):
        return self.move_piece(0, 1)
    
    def move_left(self):
        self.move_piece(-1, 0)
    
    def move_right(self):
        self.move_piece(1, 0)

    def drop_down(self):
        while self.move_down():
            if not self.stop_check():
                break
    
    def rotate(self):
        old_blocks = self.current_piece.blocks
        self.current_piece.rotate()
        if self.move_check():
            self.current_piece.blocks = old_blocks

    def run(self):
        timer = fpstimer.FPSTimer(60)
        while not self.game_over:
            self.handle_input()
            self.step()
            cv2.imshow("Tetris", self.render()), cv2.waitKey(1)
            timer.sleep()
        return(self.score)
    
    def render(self):
        # if self.progress >= levels[self.level]["gravity"] or force_render:
        img = self.draw_board()
        self.draw_piece(img)
        # img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        img = cv2.resize(img, (dimensions[0]*upscaling, dimensions[1]*upscaling), interpolation = cv2.INTER_NEAREST)
        self.draw_grid(img)
        img = cv2.putText(img, f"Score: {self.score}", (0,upscaling ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (125,23,33), 1)
        img = cv2.putText(img, f"Level: {self.level}", (0,upscaling*2 ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (125,23,33), 1)
        return img

    def step(self):
        self.handle_actions()
        if self.progress >= levels[self.level]["gravity"]:
            self.stop_check()
            self.move_down()
            self.progress = 0
        self.find_line()
        self.progress += 1
        self.ticks += 1

class Tetromino(object):
    def __init__(self, blocks, color, center, id):
        self.x = dimensions[0]//2-2
        self.y = 0
        self.color = color
        self.center = center
        self.blocks = blocks
        self.id = id
        # self.move(1, 0)
    
    def move(self, dx, dy):
        self.x += dx
        self.y += dy
    
    def get_positions(self):
        new_blocks = []
        for block in self.blocks:
            if self.y + block[1] >= 0:
                new_blocks.append([self.x + block[0], self.y + block[1]])
            else:
                new_blocks.append([self.x + block[0], 0])
        return new_blocks

    """rotates all blocks clockwise around the center"""
    def rotate(self):
        if self.color == (0, 255, 255):
            return
        
        new_blocks = []
        for block in self.blocks:
            new_blocks.append([int(self.center[0] - block[1] + self.center[1]), int(self.center[1] + block[0] - self.center[0])])
        self.blocks = new_blocks

if __name__ == '__main__':
    tetris = Tetris()
    tetris.start()