from pylabrobot.resources.coordinate import Coordinate
from pylabrobot.resources.hamilton.hamilton_decks import HamiltonDeck, _RAILS_WIDTH



class VantageDeck(HamiltonDeck):
  """ A Hamilton Vantage deck. """

  def __init__(self, size: float):
    """ Create a new Vantage deck of the given size.

    TODO: parameters for setting up the Entry Exit module, waste, etc.

    Args:
      size: The size of the deck to create. Must be 1.3 or 2.0 (meters).
    """

    # Unfortunately, float is not supported as a Literal type, so we have to use a runtime check.
    if size == 1.3:
      # Curiously stored in ML_STAR2.deck in HAMILTON\\Config after editing the deck to 1.3m using
      # the HxConfigEditor.
      super().__init__(
        num_rails=54,
        size_x=1237.5,
        size_y=653.5,
        size_z=900.0,
      )
      self.size = 1.3
    elif size == 2.0:
      raise NotImplementedError("2.0m Vantage decks are not yet supported.")
    else:
      raise ValueError(f"Invalid deck size: {size}")

  def rails_to_location(self, rails: int) -> Coordinate:
    x = 32.5 + (rails - 1) * _RAILS_WIDTH
    return Coordinate(x=x, y=63, z=100)

  def serialize(self) -> dict:
    return {
      **super().serialize(),
      "size": self.size,
    }
