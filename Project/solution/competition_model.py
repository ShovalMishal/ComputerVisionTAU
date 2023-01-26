"""Define your architecture here."""
import torch

from Project.solution.models import get_efficient_model


def my_competition_model():
    """Override the model initialization here.

    Do not change the model load line.
    """
    # initialize your model:
    model = get_efficient_model()
    # load your model using exactly this line (don't change it):
    model.load_state_dict(torch.load('checkpoints/competition_model.pt')['model'])
    return model
