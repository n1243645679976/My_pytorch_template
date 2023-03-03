import torch
import sys


def get_BTL_Tree_result(winning_matrix, round=200, tolerance=1e-4, num_players=None):
    if num_players == None:
        num_players = winning_matrix.shape[0]
    prep = torch.ones(num_players).float()
    w = winning_matrix.sum(dim=1)
    
    for _ in range(round):
        p = torch.zeros(prep.shape)
        for i in range(num_players):
            denominator = w[i]
            numerator = 0
            for j in range(num_players):
                numerator += (winning_matrix[i][j] + winning_matrix[j][i]) / (prep[i] + prep[j])
            p[i] = denominator / numerator
            if denominator == 0:
                p[i] = 0
        p /= p.sum()
        if torch.mean(torch.abs(prep - p)) < tolerance:
            break
#        print(_, torch.mean(torch.abs(prep - p)))
        prep = p
    return p


if __name__ == '__main__':
    winning_matrix = torch.tensor([[0,2,0,1], [3,0,5,0],[0,3,0,1],[4,0,3,0]]).float()
    print(get_BTL_Tree_result(winning_matrix))
