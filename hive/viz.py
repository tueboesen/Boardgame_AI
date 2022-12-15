from math import cos, sin, pi, radians
import matplotlib.pyplot as plt
import torch


def convert_board_to_indices(board):
    indices = board.nonzero()
    indices = indices.tolist()
    if type(indices) != list:
        indices = [indices]
    return indices

def shift_indices(indices):
    pass


def draw_hex(q,r):
    hex_coords = qr_to_hex_coords(q, r)
    xs, ys = zip(*(hex_coords+[hex_coords[0]]))
    # plt.plot(xs,ys,'b')
    ax = plt.gca()
    ax.fill(xs, ys, "b")

def qr_to_xy(q,r,hex_radius=20,x_offset=50,y_offset=50):
    x = x_offset + (2 * hex_radius) * r + hex_radius * q
    y = y_offset + (1.75 * hex_radius) * q
    return x, y

def qr_to_hex_coords(q, r, hex_radius=20, offset=3):
    x,y = qr_to_xy(q,r,hex_radius=hex_radius)
    hex_coords = [(x + (hex_radius + offset) * cos(radians(90) + 2 * pi * _ / 6),
                   y + (hex_radius + offset) * sin(radians(90) + 2 * pi * _ / 6))
                     for _ in range(6)]
    return hex_coords


def draw_board(board_state,types=None):
    plt.figure()
    # fig, axs = plt.subplots()

    indices = convert_board_to_indices(board_state)
    # indices = shift_indices(indices)
    for idx in indices:
        draw_hex(*idx)
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    ax.invert_yaxis()

    plt.pause(1)
