from typing import Any, Dict, Union, List, Sequence, cast
from pylabrobot.machine import MachineFrontend, need_setup_finished
from .backend import PowderDispenserBackend, PowderDispense
from pylabrobot.resources import (
  Deck,
  Resource,
  Powder
)

class PowderDispenser(MachineFrontend):
  """
  The front end for powder dispensers. Powder dispensers are devices that can dispense powder
  into containers such as mtp's placed on a deck.

  Here's an example of how to use this class:
  >>> pd = PowderDispenser(backend=YourPowderDispenserBackend(), deck=Deck())
  >>> pd.setup()
  >>> result = pd.dispense_powder(plate["A1"], powders='NaCl', amount=0.005)
  >>> print(result)
  {'actual_amount': 0.005012, ...}
  """

  def __init__(self, backend: PowderDispenserBackend, deck: Deck) -> None:
    MachineFrontend.__init__(self, backend=backend)
    self.backend: PowderDispenserBackend = backend
    self.deck = deck

  @need_setup_finished
  async def dispense(
    self,
    resources: Union[Resource, Sequence[Resource]],
    powders: Union[Powder, Sequence[Powder]],
    amounts: Union[float, Sequence[float]],
    **backend_kwargs
  ) -> List[Dict[str, Any]]:
    """
    Dispense powders into containers with specified amounts and tolerances.

    Args:
      resources (Union[Resource, Sequence[Resource]]): The target resources
        into which the dispenses should happen. Usually would be a well or a vial.
      powders (Union[Powder, Sequence[Powder]]): The powders to dispense.
      amounts (Union[float, Sequence[float]]): The amounts of powders to dispense. In mg, S.I.
      **backend_kwargs: Additional keyword arguments to be passed to the backend.

    Returns:
      List[Dict[str, Any]]: A list of dictionaries containing information about the dispensed
        powders. Should always contain the key 'actual_amount' with the actual amount
        that was dispensed.

    Raises:
      AssertionError: If the lengths of `amounts` and `powders` do not match.
    """
    if isinstance(resources, Resource):
      resources = [resources]

    if isinstance(powders, Powder):
      powders = [powders]

    if isinstance(amounts, float):
      amounts = [amounts]

    assert len(amounts) == len(powders) == len(resources)

    powder_dispenses = [
        PowderDispense(resource=r, powder=p, amount=a)
        for r, p, a in zip(
          resources,
          powders,
          amounts,
        )
      ]

    result = await self.backend.dispense(powder_dispenses, **backend_kwargs)

    return cast(List[Dict[str, Any]], result)

  async def setup(self):
    """ Setup the powder dispenser. This method should be called before any dispensing actions. """
    await super().setup()
