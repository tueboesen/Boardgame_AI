"""
These are implementation tests, that tests that all the games are playable with the player types they are supposed to support.

In order to add a game to the tests, the game should be added to games_available.py

"""
import copy
import itertools

import pytest

from Templates.viz import Vizualization
from games_available import GAMES_AVAILABLE
from src.arena import Arena
from src.mcts import MCTS, Node
from src.nnet import NNetWrapper
from src.players import PlayerTemplate, PLAYMODES


def determine_available_players(gc, playmodes=PLAYMODES):
    available_players = []
    for playmode in playmodes:
        playable = True
        args = []
        for req in playmode['req']:
            key,baseclass = req
            if key in gc and isinstance(gc[key], baseclass):
                args.append(gc[key])
            else:
                playable = False
        if playable:
            player = (playmode['fnc'],args)
            available_players.append(player)
    return available_players


def build_game_components(game_ref):
    assert 'name' in game_ref, f"{game_ref} does not have a name."
    assert 'game' in game_ref, f"{game_ref['name']} does not have a game key"
    gc = {'name': game_ref['name']}
    g = game_ref['game'][0]()
    gc['game'] = g

    key = 'nn'
    if key in game_ref:
        assert len(game_ref[key]) > 1, f'{game_ref["name"]} "nn" needs to have at least two elements (the neural network, and the arguments that goes into it)'
        nn_fnc = game_ref[key][0]
        nn_args = game_ref[key][1]
        nnet = nn_fnc(g,nn_args)
        model = NNetWrapper(g,nnet,nn_args)
        gc['nn_model'] = model
        if len(game_ref[key]) > 2:
            mcts_args = game_ref[key][2]
        else:
            mcts_args = nn_args
        mcts = MCTS(model,mcts_args)
        gc['mcts'] = mcts

    key = 'viz'
    if key in game_ref:
        viz = game_ref[key][0](g)
        gc['viz'] = viz

    return gc


def create_player_teams(nplayers,players_building_blocks):
    """
    Given a game and a list of playmodes, this function creates all types of players
    """
    combinations = itertools.combinations_with_replacement(players_building_blocks, nplayers)
    teams_building_blocks = list(combinations)
    teams = []
    for team_bb in teams_building_blocks:
        team = []
        for player_bb in team_bb:
            if (len(player_bb[1]) > 0) and isinstance(player_bb[1][0],Vizualization):
                player = player_bb[0](*copy.copy(player_bb[1]))
            else:
                player = player_bb[0](*copy.deepcopy(player_bb[1]))
            team.append(player)
        teams.append(team)
    return teams

def build_playable_games():
    for game_ref in GAMES_AVAILABLE:
        game_components = build_game_components(game_ref)
        available_players = determine_available_players(game_components)
        teams = create_player_teams(game_components['game'].number_of_players, available_players)
        for players in teams:
            # this can be simplified via get method if game_components is just a dict
            viz = game_components['viz'] if 'viz' in game_components else None
            yield players, game_components['game'], viz


@pytest.mark.parametrize("players,game,viz", build_playable_games(), ids=str)
def test_play_all_games(players, game, viz):
    if viz:
        arena = Arena(players, game, viz)
    else:
        arena = Arena(players, game)
    arena.playGames(2)

