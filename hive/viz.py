#
#
#
# def convert_board_to_indices(board):
#     indices = board.nonzero()
#     indices = indices.tolist()
#     if type(indices) != list:
#         indices = [indices]
#     return indices
#
# def shift_indices(indices):
#     pass
#
#
# def draw_hex(q,r,color='r',text='Q',ax=None):
#     hex_coords, x, y = qr_to_hex_coords(q, r)
#     xs, ys = zip(*(hex_coords+[hex_coords[0]]))
#     # plt.plot(xs,ys,'b')
#     # ax = plt.gca()
#     ax.fill(xs, ys, color)
#     if text is not None:
#         ax.text(x, y, text, fontsize=10, horizontalalignment='center', verticalalignment='center',)
#         # ax.text(x, y, text,
#         #         horizontalalignment='center',
#         #         verticalalignment='center',
#         #         transform=ax.transAxes)
#
# def qr_to_xy(q,r,hex_radius=20,x_offset=50,y_offset=50):
#     x = x_offset + (2 * hex_radius) * r + hex_radius * q
#     y = y_offset + (1.75 * hex_radius) * q
#     return x, y
#
# def qr_to_hex_coords(q, r, hex_radius=20, offset=3):
#     x,y = qr_to_xy(q,r,hex_radius=hex_radius)
#     hex_coords = [(x + (hex_radius + offset) * cos(radians(90) + 2 * pi * _ / 6),
#                    y + (hex_radius + offset) * sin(radians(90) + 2 * pi * _ / 6))
#                      for _ in range(6)]
#     return hex_coords, x,y
#
#
# def draw_board_state(board_state,types=None):
#     plt.figure()
#     # fig, axs = plt.subplots()
#
#     indices = convert_board_to_indices(board_state)
#     # indices = shift_indices(indices)
#     for idx in indices:
#         draw_hex(*idx)
#     ax = plt.gca()
#     ax.set_aspect('equal', adjustable='box')
#     ax.invert_yaxis()
#
#     plt.pause(1)
#
# def draw_board(board):
#
#     f, (ax1, ax2) = plt.subplots(1, 2)
#     for hive in board.hives:
#         color = "b" if hive.white else "r"
#         for i, (id,in_play,level,qr) in enumerate(hive):
#             qr_viz = hive.qr_viz[i]
#             if in_play and level==0:
#                 symbol = piece_symbol(id)
#                 if symbol == 'b' and hive.pieces_under[i] != 0:
#                     text = symbol + f' +{hive.pieces_under[i]}'
#                 else:
#                     text = symbol
#                 draw_hex(*qr,color=color,text=text,ax=ax1)
#                 draw_hex(*qr_viz, color=color, text=text, ax=ax2)
#     # ax = plt.gca()
#     ax1.title.set_text('Canon representation')
#     ax1.title.set_text('Viz')
#     ax1.set_aspect('equal', adjustable='box')
#     ax2.set_aspect('equal', adjustable='box')
#     ax1.axis('off')
#     ax2.axis('off')
#     # ax1.invert_yaxis()
#     plt.pause(1)
#     # print('done')
#     # draw_board_state(board.board_state)
#
#
