""" Defines LiquidHandler class, the coordinator for liquid handling operations. """

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import math
import numbers
import threading
import time
from typing import Any, Callable, Dict, Union, Optional, List, Sequence, Set, Tuple
import warnings
import itertools

from pylabrobot.machine import MachineFrontend, need_setup_finished
from pylabrobot.liquid_handling.strictness import Strictness, get_strictness
from pylabrobot.plate_reading import PlateReader
from pylabrobot.resources import (
  Container,
  Deck,
  Resource,
  ResourceStack,
  Coordinate,
  CarrierSite,
  Lid,
  Plate,
  Tip,
  TipRack,
  TipSpot,
  Well,
  TipTracker,
  does_tip_tracking,
  does_volume_tracking
)
from pylabrobot.resources.liquid import Liquid
from pylabrobot.resources.errors import NoTipError
from pylabrobot.utils.list import expand

from .backends import LiquidHandlerBackend
from .standard import (
  Pickup,
  PickupTipRack,
  Drop,
  DropTipRack,
  Aspiration,
  AspirationPlate,
  Dispense,
  DispensePlate,
  Move,
  GripDirection
)

logger = logging.getLogger("pylabrobot")


class LiquidHandler(MachineFrontend):
  """
  Front end for liquid handlers.

  This class is the front end for liquid handlers; it provides a high-level interface for
  interacting with liquid handlers. In the background, this class uses the low-level backend (
  defined in `pyhamilton.liquid_handling.backends`) to communicate with the liquid handler.
  """

  def __init__(self, backend: LiquidHandlerBackend, deck: Deck):
    """ Initialize a LiquidHandler.

    Args:
      backend: Backend to use.
      deck: Deck to use.
    """

    super().__init__(backend=backend)

    self.backend: LiquidHandlerBackend = backend
    self._picked_up_tips96: Optional[TipRack] = None # TODO: replace with tracker.

    self.deck = deck
    self.deck.resource_assigned_callback_callback = self.resource_assigned_callback
    self.deck.resource_unassigned_callback_callback = self.resource_unassigned_callback

    self.head: Dict[int, TipTracker] = {}


  async def setup(self):
    """ Prepare the robot for use. """

    if self.setup_finished:
      raise RuntimeError("The setup has already finished. See `LiquidHandler.stop`.")

    await self.backend.setup()

    self.head = {c: TipTracker(thing=f"Channel {c}") for c in range(self.backend.num_channels)}

    self.resource_assigned_callback(self.deck)
    for resource in self.deck.children:
      self.resource_assigned_callback(resource)

    await super().setup()

  def save_state(self, filename: str):
    """ Save the state of the liquid handler (including the deck) to a file. """

    head_state = {channel: tracker.serialize() for channel, tracker in self.head.items()}
    deck_state = self.deck.serialize_state()
    state = {"head_state": head_state, "deck_state": deck_state}
    with open(filename, "w", encoding="utf-8") as f:
      f.write(json.dumps(state, indent=2))

  def load_state(self, filename: str):
    """ Load the liquid handler state from a file. """

    with open(filename, "r", encoding="utf-8") as f:
      state = json.load(f)
    head_state = state["head_state"]
    deck_state = state["deck_state"]

    for channel_id, state in head_state.items():
      # convert head state keys (channel ids) back to integers.
      self.head[int(channel_id)].load_state(state)
    self.deck.load_state(deck_state)

  def update_head_state(self, state: Dict[int, Optional[Tip]]):
    """ Update the state of the liquid handler head.

    All keys in `state` must be valid channels. Channels for which no key is specified will keep
    their current state.

    Args:
      state: A dictionary mapping channels to tips. If a channel is mapped to None, that channel
        will have no tip.
    """

    assert set(state.keys()).issubset(set(self.head.keys())), "Invalid channel."

    for channel, tip in state.items():
      if tip is None:
        if self.head[channel].has_tip:
          self.head[channel].remove_tip()
      else:
        self.head[channel].add_tip(tip)

  def clear_head_state(self):
    """ Clear the state of the liquid handler head. """

    self.update_head_state({c: None for c in self.head.keys()})

  def _run_async_in_thread(self, func, *args, **kwargs):
    def callback(*args, **kwargs):
      loop = asyncio.new_event_loop()
      asyncio.set_event_loop(loop)
      loop.run_until_complete(func(*args, **kwargs))

    t = threading.Thread(target=callback, args=args, kwargs=kwargs)
    t.start()
    t.join()

  def resource_assigned_callback(self, resource: Resource):
    self._run_async_in_thread(self.backend.assigned_resource_callback, resource)

  def resource_unassigned_callback(self, resource: Resource):
    self._run_async_in_thread(self.backend.unassigned_resource_callback, resource.name)

  def unassign_resource(self, resource: Union[str, Resource]): # TODO: remove this.
    """ Unassign an assigned resource.

    Args:
      resource: The resource to unassign.

    Raises:
      KeyError: If the resource is not currently assigned to this liquid handler.
    """

    if isinstance(resource, Resource):
      resource = resource.name

    r = self.deck.get_resource(resource)
    if r is None:
      raise KeyError(f"Resource '{resource}' is not assigned to this liquid handler.")
    r.unassign()

  def get_resource(self, name: str) -> Resource:
    """ Find a resource on the deck of this liquid handler. Also see :meth:`~Deck.get_resource`.

    Args:
      name: name of the resource.

    Returns:
      The resource with the given name, or None if not found.
    """

    return self.deck.get_resource(name)

  def summary(self):
    """ Prints a string summary of the deck layout.  """

    print(self.deck.summary())

  def _assert_positions_unique(self, positions: List[str]):
    """ Returns whether all items in `positions` are unique where they are not `None`.

    Args:
      positions: List of positions.
    """

    not_none = [p for p in positions if p is not None]
    if len(not_none) != len(set(not_none)):
      raise ValueError("Positions must be unique.")

  def _assert_resources_exist(self, resources: Sequence[Resource]):
    """ Checks that each resource in `resources` is assigned to the deck.

    Args:
      resources: List of resources.

    Raises:
      ValueError: If a resource is not assigned to the deck.
    """

    for resource in resources:
      # see of top parent of resource is the deck (i.e. resource is assigned to deck)
      while resource.parent is not None:
        resource = resource.parent
      if resource != self.deck:
        raise ValueError(f"Resource named '{resource.name}' not found on deck.")

  def _check_args(
    self,
    method: Callable,
    backend_kwargs: Dict[str, Any],
    default: Set[str]
  ) -> Set[str]:
    """ Checks that the arguments to `method` are valid.

    Args:
      method: Method to check.
      backend_kwargs: Keyword arguments to `method`.

    Raises:
      TypeError: If the arguments are invalid.

    Returns:
      The set of arguments that need to be removed from `backend_kwargs` before passing to `method`.
    """

    default_args = default.union({"self"})

    sig = inspect.signature(method)
    args = {arg: param for arg, param in sig.parameters.items() if arg not in default_args}
    vars_keyword = {arg for arg, param in sig.parameters.items() # **kwargs
                    if param.kind == inspect.Parameter.VAR_KEYWORD}
    args = {arg: param for arg, param in args.items() # keep only *args and **kwargs
            if param.kind not in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}}
    non_default = {arg for arg, param in args.items() if param.default == inspect.Parameter.empty}

    strictness = get_strictness()

    backend_kws = set(backend_kwargs.keys())

    missing = non_default - backend_kws
    if len(missing) > 0:
      raise TypeError(f"Missing arguments to backend.{method.__name__}: {missing}")

    extra = backend_kws - set(args.keys())

    if len(extra) > 0 and len(vars_keyword) == 0:
      if strictness == Strictness.STRICT:
        raise TypeError(f"Extra arguments to backend.{method.__name__}: {extra}")
      elif strictness == Strictness.WARN:
        warnings.warn(f"Extra arguments to backend.{method.__name__}: {extra}")
      else:
        logger.debug("Extra arguments to backend.%s: %s", method.__name__, extra)

    return extra

  def _make_sure_channels_exist(self, channels: List[int]):
    """ Checks that the channels exist. """
    invalid_channels = [c for c in channels if c not in self.head]
    if not len(invalid_channels) == 0:
      raise ValueError(f"Invalid channels: {invalid_channels}")

  @need_setup_finished
  async def pick_up_tips(
    self,
    tip_spots: List[TipSpot],
    use_channels: Optional[List[int]] = None,
    offsets: Optional[Union[Coordinate, List[Optional[Coordinate]]]] = None,
    **backend_kwargs
  ):
    """ Pick up tips from a resource.

    Examples:
      Pick up all tips in the first column.

      >>> lh.pick_up_tips(tips_resource["A1":"H1"])

      Pick up tips on odd numbered rows, skipping the other channels.

      >>> lh.pick_up_tips(tips_resource["A1", "C1", "E1", "G1"],use_channels=[0, 2, 4, 6])

      Pick up tips from different tip resources:

      >>> lh.pick_up_tips(tips_resource1["A1"] + tips_resource2["B2"] + tips_resource3["C3"])

      Picking up tips with different offsets:

      >>> lh.pick_up_tips(
      ...   tip_spots=tips_resource["A1":"C1"],
      ...   offsets=[
      ...     Coordinate(0, 0, 0), # A1
      ...     Coordinate(1, 1, 1), # B1
      ...     Coordinate.zero() # C1
      ...   ]
      ... )

    Args:
      tip_spots: List of tip spots to pick up tips from.
      use_channels: List of channels to use. Index from front to back. If `None`, the first
        `len(channels)` channels will be used.
      offsets: List of offsets for each channel, a translation that will be applied to the tip
        drop location. If `None`, no offset will be applied.
      backend_kwargs: Additional keyword arguments for the backend, optional.

    Raises:
      RuntimeError: If the setup has not been run. See :meth:`~LiquidHandler.setup`.

      ValueError: If the positions are not unique.

      HasTipError: If a channel already has a tip.

      NoTipError: If a spot does not have a tip.
    """

    self._assert_resources_exist(tip_spots)

    offsets = expand(offsets, len(tip_spots))

    if use_channels is None:
      use_channels = list(range(len(tip_spots)))

    self._make_sure_channels_exist(use_channels)

    assert len(tip_spots) == len(offsets) == len(use_channels), \
      "Number of tips and offsets and use_channels must be equal."

    tips: List[Tip] = []
    for tip_spot in tip_spots:
      tips.append(tip_spot.get_tip())

    pickups = [Pickup(resource=tip_spot, offset=offset, tip=tip)
               for tip_spot, offset, tip in zip(tip_spots, offsets, tips)]

    for channel, op in zip(use_channels, pickups):
      if does_tip_tracking() and not op.resource.tracker.is_disabled:
        op.resource.tracker.remove_tip()
      if not does_tip_tracking() and self.head[channel].has_tip:
        self.head[channel].remove_tip() # override the tip if a tip exists
      self.head[channel].add_tip(op.tip, origin=op.resource, commit=False)

    extras = self._check_args(self.backend.pick_up_tips, backend_kwargs,
      default={"ops", "use_channels"})
    for extra in extras:
      del backend_kwargs[extra]

    try:
      await self.backend.pick_up_tips(ops=pickups, use_channels=use_channels, **backend_kwargs)
    except:
      for channel, op in zip(use_channels, pickups):
        if does_tip_tracking() and not op.resource.tracker.is_disabled:
          op.resource.tracker.rollback()
        self.head[channel].rollback()
      raise
    else:
      for channel, op in zip(use_channels, pickups):
        if does_tip_tracking() and not op.resource.tracker.is_disabled:
          op.resource.tracker.commit()
        self.head[channel].commit()

  @need_setup_finished
  async def drop_tips(
    self,
    tip_spots: List[Union[TipSpot, Resource]],
    use_channels: Optional[List[int]] = None,
    offsets: Optional[Union[Coordinate, List[Optional[Coordinate]]]] = None,
    allow_nonzero_volume: bool = False,
    **backend_kwargs
  ):
    """ Drop tips to a resource.

    Examples:
      Dropping tips to the first column.

      >>> lh.pick_up_tips(tip_rack["A1:H1"])

      Dropping tips with different offsets:

      >>> lh.drop_tips(
      ...   channels=tips_resource["A1":"C1"],
      ...   offsets=[
      ...     Coordinate(0, 0, 0), # A1
      ...     Coordinate(1, 1, 1), # B1
      ...     Coordinate.zero() # C1
      ...   ]
      ... )

    Args:
      tips: Tip resource locations to drop to.
      use_channels: List of channels to use. Index from front to back. If `None`, the first
        `len(channels)` channels will be used.
      offsets: List of offsets for each channel, a translation that will be applied to the tip
        pickup location. If `None`, no offset will be applied.
      allow_nonzero_volume: If `True`, the tip will be dropped even if its volume is not zero (there
        is liquid in the tip). If `False`, a RuntimeError will be raised if the tip has nonzero
        volume.
      backend_kwargs: Additional keyword arguments for the backend, optional.

    Raises:
      RuntimeError: If the setup has not been run. See :meth:`~LiquidHandler.setup`.

      ValueError: If no channel will pick up a tip, in other words, if all channels are `None` or
        if the list of channels is empty.

      ValueError: If the positions are not unique.

      NoTipError: If a channel does not have a tip.

      HasTipError: If a spot already has a tip.
    """

    self._assert_resources_exist(tip_spots)

    offsets = expand(offsets, len(tip_spots))

    if use_channels is None:
      use_channels = list(range(len(tip_spots)))

    self._make_sure_channels_exist(use_channels)

    tips = []
    for channel in use_channels:
      tip = self.head[channel].get_tip()
      if tip.tracker.get_used_volume() > 0 and not allow_nonzero_volume:
        raise RuntimeError(f"Cannot drop tip with volume {tip.tracker.get_used_volume()}")
      tips.append(tip)

    assert len(tip_spots) == len(offsets) == len(use_channels) == len(tips), \
      "Number of channels and offsets and use_channels and tips must be equal."

    drops = [Drop(resource=tip_spot, offset=offset, tip=tip)
             for tip_spot, tip, offset in zip(tip_spots, tips, offsets)]

    for channel, op in zip(use_channels, drops):
      if does_tip_tracking() and isinstance(op.resource, TipSpot) and \
          not op.resource.tracker.is_disabled:
        op.resource.tracker.add_tip(op.tip, commit=False)
      self.head[channel].remove_tip()

    extras = self._check_args(self.backend.drop_tips, backend_kwargs,
      default={"ops", "use_channels"})
    for extra in extras:
      del backend_kwargs[extra]

    try:
      await self.backend.drop_tips(ops=drops, use_channels=use_channels, **backend_kwargs)
    except:
      for channel, op in zip(use_channels, drops):
        if does_tip_tracking() and \
          (isinstance(op.resource, TipSpot) and not op.resource.tracker.is_disabled):
          op.resource.tracker.rollback()
        self.head[channel].rollback()
      raise
    else:
      for channel, op in zip(use_channels, drops):
        if does_tip_tracking() and \
          (isinstance(op.resource, TipSpot) and not op.resource.tracker.is_disabled):
          op.resource.tracker.commit()
        self.head[channel].commit()

  async def return_tips(self):
    """ Return all tips that are currently picked up to their original place.

    Examples:
      Return the tips on the head to the tip rack where they were picked up:

      >>> lh.pick_up_tips(tip_rack["A1"])
      >>> lh.return_tips()

    Raises:
      RuntimeError: If no tips have been picked up.
    """

    tip_spots: List[TipSpot] = []
    channels: List[int] = []

    for channel, tracker in self.head.items():
      if tracker.has_tip:
        origin = tracker.get_tip_origin()
        if origin is None:
          raise RuntimeError("No tip origin found.")
        tip_spots.append(origin)
        channels.append(channel)

    if len(tip_spots) == 0:
      raise RuntimeError("No tips have been picked up.")

    return await self.drop_tips(tip_spots=tip_spots, use_channels=channels)

  async def discard_tips(
    self,
    use_channels: Optional[List[int]] = None,
    **backend_kwargs
  ):
    """ Permanently discard tips.

    Examples:
      Discarding the tips on channels 1 and 2:

      >>> lh.discard_tips(use_channels=[0, 1])

      Discarding all tips currently picked up:

      >>> lh.discard_tips()

    Args:
      use_channels: List of channels to use. Index from front to back. If `None`, all that have
        tips will be used.
      backend_kwargs: Additional keyword arguments for the backend, optional.
    """

    # Different default value from drop_tips: here we factor in the tip tracking.
    if use_channels is None:
      use_channels = [c for c, t in self.head.items() if t.has_tip]

    n = len(use_channels)

    if n == 0:
      raise RuntimeError("No tips have been picked up and no channels were specified.")

    trash = self.deck.get_trash_area()
    offsets = trash.get_2d_center_offsets(n=n)

    return await self.drop_tips(
        tip_spots=[trash]*n,
        use_channels=use_channels,
        offsets=offsets,
        **backend_kwargs)

  @need_setup_finished
  async def aspirate(
    self,
    resources: Union[Container, Sequence[Container]],
    vols: Union[List[float], float],
    use_channels: Optional[List[int]] = None,
    flow_rates: Optional[Union[float, List[Optional[float]]]] = None,
    end_delay: float = 0,
    offsets: Union[Optional[Coordinate], Sequence[Optional[Coordinate]]] = None,
    liquid_height: Union[Optional[float], List[Optional[float]]] = None,
    **backend_kwargs
  ):
    """ Aspirate liquid from the specified wells.

    Examples:
      Aspirate a constant amount of liquid from the first column:

      >>> lh.aspirate(plate["A1:H1"], 50)

      Aspirate an linearly increasing amount of liquid from the first column:

      >>> lh.aspirate(plate["A1:H1"], range(0, 500, 50))

      Aspirate arbitrary amounts of liquid from the first column:

      >>> lh.aspirate(plate["A1:H1"], [0, 40, 10, 50, 100, 200, 300, 400])

      Aspirate liquid from wells in different plates:

      >>> lh.aspirate(plate["A1"] + plate2["A1"] + plate3["A1"], 50)

      Aspirating with a 10mm z-offset:

      >>> lh.aspirate(plate["A1"], vols=50, offsets=[Coordinate(0, 0, 10)])

      Aspirate from a blue bucket (big container), with the first 4 channels (which will be
      spaced equally apart):

      >>> lh.aspirate(blue_bucket, vols=50, use_channels=[0, 1, 2, 3])

    Args:
      resources: A list of wells to aspirate liquid from. Can be a single resource, or a list of
        resources. If a single resource is specified, all channels will aspirate from the same
        resource.
      vols: A list of volumes to aspirate, one for each channel. If `vols` is a single number, then
        all channels will aspirate that volume.
      use_channels: List of channels to use. Index from front to back. If `None`, the first
        `len(wells)` channels will be used.
      flow_rates: the aspiration speed. In ul/s. If `None`, the backend default will be used.
      end_delay: The delay after the last aspiration in seconds, optional. This is useful for when
        the tips used in the aspiration are dripping.
      offsets: List of offsets for each channel, a translation that will be applied to the
        aspiration location. If `None`, no offset will be applied.
      liquid_height: The height of the liquid in the well, in mm.
      backend_kwargs: Additional keyword arguments for the backend, optional.

    Raises:
      RuntimeError: If the setup has not been run. See :meth:`~LiquidHandler.setup`.

      ValueError: If all channels are `None`.
    """

    # Start with computing the locations of the aspirations. Can either be a single resource, in
    # which case all channels will aspirate from there, or a list of resources.
    if isinstance(resources, Resource): # if single resource, space channels evenly
      if use_channels is None:
        use_channels = [0]

      self._make_sure_channels_exist(use_channels)

      n = len(use_channels)

      # If offsets is supplied, make sure it is a list of the correct length. If it is not in this
      # format, raise an error. If it is not supplied, make it a list of the correct length by
      # spreading channels across the resource evenly.
      if offsets is not None:
        if not isinstance(offsets, list) or len(offsets) != n:
          raise ValueError("Number of offsets must match number of channels used when aspirating "
                           "from a resource.")
      else:
        offsets = resources.get_2d_center_offsets(n=n)

      resources = [resources] * n
    else:
      if len(resources) == 0:
        raise ValueError("No channels specified")
      self._assert_resources_exist(resources)
      n = len(resources)

      for resource in resources:
        if isinstance(resource.parent, Plate) and resource.parent.has_lid():
          raise ValueError("Aspirating from plate with lid")

      if use_channels is None:
        use_channels = list(range(len(resources)))

      self._make_sure_channels_exist(use_channels)

      offsets = expand(offsets, n)

    vols = expand(vols, n)
    flow_rates = expand(flow_rates, n)
    liquid_height = expand(liquid_height, n)
    tips = [self.head[channel].get_tip() for channel in use_channels]

    assert len(vols) == len(offsets) == len(flow_rates) == len(liquid_height)

    # liquid(s) for each channel. If volume tracking is disabled, use None as the liquid.
    liquids: List[List[Tuple[Optional[Liquid], float]]] = []
    for r, vol in zip(resources, vols):
      if r.tracker.is_disabled or not does_volume_tracking():
        liquids.append([(None, vol)])
      else:
        liquids.append(r.tracker.get_liquids(top_volume=vol))

    aspirations = [Aspiration(resource=r, volume=v, offset=o, flow_rate=fr, liquid_height=lh, tip=t,
                              blow_out_air_volume=0, liquids=lvs)
                   for r, v, o, fr, lh, t, lvs in
                    zip(resources, vols, offsets, flow_rates, liquid_height, tips, liquids)]

    for op in aspirations:
      if does_volume_tracking():
        if not op.resource.tracker.is_disabled:
          op.resource.tracker.remove_liquid(op.volume)
        for liquid, volume in reversed(op.liquids):
          op.tip.tracker.add_liquid(liquid=liquid, volume=volume)

    extras = self._check_args(self.backend.aspirate, backend_kwargs,
      default={"ops", "use_channels"})
    for extra in extras:
      del backend_kwargs[extra]

    try:
      await self.backend.aspirate(ops=aspirations, use_channels=use_channels, **backend_kwargs)
    except:
      for op in aspirations:
        if not op.resource.tracker.is_disabled:
          op.resource.tracker.rollback()
          op.tip.tracker.rollback()
      raise
    else:
      for op in aspirations:
        if not op.resource.tracker.is_disabled:
          op.resource.tracker.commit()
          op.tip.tracker.commit()
      for tracker in self.head.values():
        tracker.commit()

    if end_delay > 0:
      time.sleep(end_delay)

  @need_setup_finished
  async def dispense(
    self,
    resources: Union[Container, Sequence[Container]],
    vols: Union[List[float], float],
    use_channels: Optional[List[int]] = None,
    flow_rates: Optional[Union[float, List[Optional[float]]]] = None,
    end_delay: float = 0,
    offsets: Union[Optional[Coordinate], Sequence[Optional[Coordinate]]] = None,
    liquid_height: Union[Optional[float], List[Optional[float]]] = None,
    **backend_kwargs
  ):
    """ Dispense liquid to the specified channels.

    Examples:
      Dispense a constant amount of liquid to the first column:

      >>> lh.dispense(plate["A1:H1"], 50)

      Dispense an linearly increasing amount of liquid to the first column:

      >>> lh.dispense(plate["A1:H1"], range(0, 500, 50))

      Dispense arbitrary amounts of liquid to the first column:

      >>> lh.dispense(plate["A1:H1"], [0, 40, 10, 50, 100, 200, 300, 400])

      Dispense liquid to wells in different plates:

      >>> lh.dispense((plate["A1"], 50), (plate2["A1"], 50), (plate3["A1"], 50))

      Dispensing with a 10mm z-offset:

      >>> lh.dispense(plate["A1"], vols=50, offsets=[Coordinate(0, 0, 10)])

      Dispense a blue bucket (big container), with the first 4 channels (which will be spaced
      equally apart):

      >>> lh.dispense(blue_bucket, vols=50, use_channels=[0, 1, 2, 3])

    Args:
      wells: A list of resources to dispense liquid to. Can be a list of resources, or a single
        resource, in which case all channels will dispense to that resource.
      vols: A list of volumes to dispense, one for each channel, or a single volume to dispense to
        all channels. If `vols` is a single number, then all channels will dispense that volume. In
        units of ul.
      use_channels: List of channels to use. Index from front to back. If `None`, the first
        `len(channels)` channels will be used.
      flow_rates: the flow rates, in ul/s. If `None`, the backend default will be used.
      end_delay: The delay after the last dispense in seconds, optional. This is useful for when
        the tips used in the dispense are dripping.
      offsets: List of offsets for each channel, a translation that will be applied to the
        dispense location. If `None`, no offset will be applied.
      liquid_height: The height of the liquid in the well, in mm.
      backend_kwargs: Additional keyword arguments for the backend, optional.

    Raises:
      RuntimeError: If the setup has not been run. See :meth:`~LiquidHandler.setup`.

      ValueError: If the dispense info is invalid, in other words, when all channels are `None`.

      ValueError: If all channels are `None`.
    """

    # Start with computing the locations of the dispenses. Can either be a single resource, in
    # which case all channels will dispense to there, or a list of resources.
    if isinstance(resources, Resource): # if single resource, space channels evenly
      if use_channels is None:
        use_channels = list(range(len(vols))) if isinstance(vols, list) else [0]

      self._make_sure_channels_exist(use_channels)

      n = len(use_channels)

      # If offsets is supplied, make sure it is a list of the correct length. If it is not in this
      # format, raise an error. If it is not supplied, make it a list of the correct length by
      # spreading channels across the resource evenly.
      if offsets is not None:
        if not isinstance(offsets, list) or len(offsets) != n:
          raise ValueError("Number of offsets must match number of channels used when dispensing "
                          "to a resource.")
      else:
        offsets = resources.get_2d_center_offsets(n=n)

      resources = [resources] * n
    else:
      if len(resources) == 0:
        raise ValueError("No channels specified")
      self._assert_resources_exist(resources)
      n = len(resources)

      for resource in resources:
        if isinstance(resource.parent, Plate) and resource.parent.has_lid():
          raise ValueError("Dispensing to plate with lid")

      if use_channels is None:
        use_channels = list(range(len(resources)))

      self._make_sure_channels_exist(use_channels)

      offsets = expand(offsets, n)

    vols = expand(vols, n)
    flow_rates = expand(flow_rates, n)
    liquid_height = expand(liquid_height, n)
    tips = [self.head[channel].get_tip() for channel in use_channels]

    assert len(vols) == len(offsets) == len(flow_rates) == len(liquid_height)

    if does_volume_tracking():
      liquids = [c.get_tip().tracker.get_liquids(top_volume=vol)
                for c, vol in zip(self.head.values(), vols)]
    else:
      liquids = [[(None, vol)] for vol in vols]

    dispenses = [Dispense(resource=r, volume=v, offset=o, flow_rate=fr, liquid_height=lh, tip=t,
                          liquids=lvs, blow_out_air_volume=0) # TODO: get blow_out_air_volume
                 for r, v, o, fr, lh, t, lvs in
                  zip(resources, vols, offsets, flow_rates, liquid_height, tips, liquids)]

    for op in dispenses:
      if does_volume_tracking():
        if not op.resource.tracker.is_disabled:
          for liquid, volume in op.liquids:
            op.resource.tracker.add_liquid(liquid=liquid, volume=volume)
        op.tip.tracker.remove_liquid(op.volume)

    extras = self._check_args(self.backend.dispense, backend_kwargs,
      default={"ops", "use_channels"})
    for extra in extras:
      del backend_kwargs[extra]

    try:
      await self.backend.dispense(ops=dispenses, use_channels=use_channels, **backend_kwargs)
    except:
      for op in dispenses:
        if not op.resource.tracker.is_disabled:
          op.resource.tracker.rollback()
          op.tip.tracker.rollback()
      raise
    else:
      for op in dispenses:
        if not op.resource.tracker.is_disabled:
          op.resource.tracker.commit()
          op.tip.tracker.commit()

    if end_delay > 0:
      time.sleep(end_delay)


  async def transfer(
    self,
    sources: Union[Well, List[Well]],
    targets: Union[Well, List[Well]],
    source_vol: Optional[Union[float, List[Optional[float]]]] = None,
    ratios: Optional[List[float]] = None,
    target_vols: Optional[Union[float, List[Optional[float]]]] = None,
    aspiration_flow_rates: Optional[Union[float, List[Optional[float]]]] = None,
    dispense_flow_rates: Optional[Union[float, List[Optional[float]]]] = None,
    use_channels: Optional[Union[int, List[Optional[int]]]] = None,
    reuse_tips: bool = False,
    use_backend_kwargs: str = "both", # can take, "only_on_aspirate", "only_on_dispense", or "both"
    **backend_kwargs
  ):
    """Transfer liquid from one well to another.
    Can transfer 1 to 1, 1 to many, many to 1, or many to many,
    based on if sources and targets are wells or lists of wells.

    Examples:

      1 to 1:
      Transfer 50 uL of liquid from the first well to the second well:

      >>> lh.transfer(plate["A1"], plate["B1"], source_vol=50)

      1 to many:
      Transfer 80 uL of liquid from the first well equally to the first column:

      >>> lh.transfer(plate["A1"], plate["A1:H1"], source_vol=80)

      Transfer 60 uL of liquid from the first well in a 1:2 ratio to 2 other wells:

      >>> lh.transfer(plate["A1"], plate["B1:C1"], source_vol=60, ratios=[2, 1])

      Transfer arbitrary volumes to the first column:

      >>> lh.transfer(plate["A1"], plate["A1:H1"], target_vols=[3, 1, 4, 1, 5, 9, 6, 2])

      many to 1:
      Transfer 50 uL of liquid from the first column to the first well:

      >>> lh.transfer(plate["A1:H1"], plate["H8"], target_vol=50)

    Args:
      source: The source well.
      targets: The target wells.
      source_vol: The volume to transfer from the source well.
      ratios: The ratios to use when transferring liquid to the target wells. If not specified, then
        the volumes will be distributed equally.
      target_vols: The volumes to transfer to the target wells.
      aspiration_flow_rate: The flow rate to use when aspirating, in ul/s. If `None`, the backend
        default will be used. Either a single flow rate for all channels, or a list of flow rates,
        one for each target well.
      dispense_flow_rates: The flow rates to use when dispensing, in ul/s. If `None`, the backend
        default will be used. Either a single flow rate for all channels, or a list of flow rates,
        one for each target well.

    Raises:
      RuntimeError: If the setup has not been run. See :meth:`~LiquidHandler.setup`.
    """
    # specify when to use backend_kwargs for aspiration and dispense
    # todo: add support for different backend_kwargs for aspiration and dispense
    assert use_backend_kwargs in ["only_on_aspirate", "only_on_dispense", "both"], ValueError("use_backend_kwargs must be one of 'only_on_aspirate', 'only_on_dispense', or 'both'")
    if use_backend_kwargs in ["both", "only_on_aspirate"]:
      aspiration_kwargs = backend_kwargs
    if use_backend_kwargs in ["both", "only_on_dispense"]:
      dispense_kwargs = backend_kwargs

<<<<<<< Updated upstream
    if isinstance(sources, Well):
      sources = [sources]

    if isinstance(targets, Well):
      targets = [targets]
=======
>>>>>>> Stashed changes


    """Case 1: Single Well to Single Well"""
    if isinstance(sources, Well) and isinstance(targets, Well):
      # For each case, we have to recalculate all of the parameters
      # source_vol, target_vol and ratios:
      assert (source_vol is not None) ^ (target_vols is not None), \
        ValueError("1 to 1: Must specify source_vol or target_vols, but not both, got", source_vol, target_vols)
      assert source_vol is None or isinstance(source_vol, numbers.Rational) or len(source_vol) == 1, \
        ValueError("1 to 1: source_vol must have only one value, got", source_vol)
      assert target_vols is None or isinstance(target_vols, numbers.Rational) or len(target_vols) == 1, \
        ValueError("1 to 1: target_vols must have only one value, got", target_vols)
      assert ratios is None, ValueError("1 to 1: Cannot specify ratios, got", ratios)

      # set source_vol equal to target_vol convert to list of length 1
      volume = source_vol if source_vol is not None else target_vols
      if isinstance(volume, float):
        volume = [volume]

<<<<<<< Updated upstream
      # ! this has to be recalculated for multiple source vols
      target_vols = [source_vol[0] * r / sum(ratios) for r in ratios]

    await self.aspirate(
      resources=[sources],
      vols=[sum(target_vols) / len(sources)] * len(sources),
      flow_rates=aspiration_flow_rate,
      **backend_kwargs)
    for target, vol in zip(targets, target_vols):
=======
      # aspiration_flow_rates: convert to list of length 1
      assert aspiration_flow_rates is None or isinstance(aspiration_flow_rates, int) or \
        len(aspiration_flow_rates) == 1, \
        ValueError("1 to 1: aspiration_flow_rates must have only one value, got", aspiration_flow_rates)
      if isinstance(aspiration_flow_rates, int):
        aspiration_flow_rates = [aspiration_flow_rates]
      # dispense_flow_rates: convert to list of length 1
      assert dispense_flow_rates is None or isinstance(dispense_flow_rates, int) or len(dispense_flow_rates) == 1, \
        ValueError("1 to 1: dispense_flow_rates must have only one value, got", dispense_flow_rates)
      if isinstance(dispense_flow_rates, int):
        dispense_flow_rates = [dispense_flow_rates]

      # use_channels: convert to 0 or int
      assert use_channels is None or isinstance(use_channels, int) or len(use_channels) == 1, \
        ValueError("1 to 1: use_channels must have only one value, got", use_channels)
      if isinstance(use_channels, list):
        use_channels = use_channels[0]
      if use_channels is None:
        use_channels = 0

      await self.aspirate(
      resources=[sources],
      vols=volume,
      flow_rates=aspiration_flow_rates,
      **aspiration_kwargs)
>>>>>>> Stashed changes
      await self.dispense(
        resources=[targets],
        vols=volume,
        flow_rates=dispense_flow_rates,
        use_channels=use_channels,
        **dispense_kwargs)
      return

    """ Case 2: Single Well to Multiple Wells """
    if isinstance(sources, Well) and isinstance(targets, list):

      # source_vol, target_vol and ratios:
      assert (source_vol is not None) ^ (target_vols is not None), \
        ValueError("1 to many: Must specify source_vol or target_vols, but not both, got", source_vol, target_vols)
      assert source_vol is None or \
        isinstance(source_vol, numbers.Rational) or \
        len(source_vol) == 1, \
        ValueError("1 to many: source_vol must have only one value, got", source_vol)
      assert target_vols is None or \
        isinstance(target_vols, numbers.Rational) or \
        len(target_vols) == len(targets), \
        ValueError("1 to many: target_vols must be a number or have same length as targets, got", target_vols)

      def sum_mixed(x: Union[float, List[int]]) -> int:
        return sum(x) if isinstance(x, list) else x

      # source_vol: get or calculate from targets and convert to list of length 1
      source_vol = source_vol if source_vol is not None else sum_mixed(target_vols)
      if isinstance(source_vol, numbers.Rational):
        source_vol = [source_vol]

      # target_vols: get or calculate from sources and convert to list (based optionally on ratios)
      if ratios is None:
          ratios = [1] * len(targets)
      if isinstance(target_vols, numbers.Rational):
        target_vols = [target_vols * r / sum(ratios) for r in ratios]
      elif target_vols is None:
        target_vols = [source_vol[0] * r / sum(ratios) for r in ratios]

      # aspiration_flow_rates: convert to list of length 1
      assert aspiration_flow_rates is None or isinstance(aspiration_flow_rates, int) or len(aspiration_flow_rates) == 1, \
        ValueError("1 to 1: aspiration_flow_rates must have only one value, got", aspiration_flow_rates)
      if isinstance(aspiration_flow_rates, int):
        aspiration_flow_rates = [aspiration_flow_rates]
      # dispense_flow_rates: convert to list of length targets
      assert dispense_flow_rates is None or \
        isinstance(dispense_flow_rates, numbers.Rational) or \
        len(dispense_flow_rates) == len(targets), \
        ValueError("1 to 1: dispense_flow_rates must be a number or a list with the same length as targets, got", dispense_flow_rates)
      if isinstance(dispense_flow_rates, numbers.Rational):
        dispense_flow_rates = [dispense_flow_rates] * len(targets)

      # use_channels: convert to 0 or int
      # TODO can handle multiple channels, this way you can keep it clean
      assert use_channels is None or isinstance(use_channels, int) or len(use_channels) == 1, \
        ValueError("1 to many: use_channels must have only one value, got", use_channels)
      if isinstance(use_channels, list):
        use_channels = use_channels[0]
      if use_channels is None:
        use_channels = 0


      await self.aspirate(
      resources=[sources],
      vols=source_vol,
      flow_rates=aspiration_flow_rates,
      **aspiration_kwargs)
      for target, vol, disp_flow in zip(targets, target_vols, dispense_flow_rates):
        await self.dispense(
          resources=[target],
          vols=vol,
          flow_rates=disp_flow,
          use_channels=[use_channels],
          **backend_kwargs)
      return


    """  Case 3: Multiple Wells to Single Well """
    if isinstance(sources, list) and isinstance(targets, Well):
      # source_vol, target_vol and ratios:
      assert (source_vol is not None) ^ (target_vols is not None), \
        ValueError("many to 1: Must specify source_vol or target_vols, but not both, got", source_vol, target_vols)
      assert source_vol is None or \
        isinstance(source_vol, numbers.Rational) or \
        len(source_vol) == len(sources), \
        ValueError("many to 1: source_vol must have same length as sources, got", source_vol)
      assert target_vols is None or \
        isinstance(target_vols, numbers.Rational) or \
        len(target_vols) == 1, \
        ValueError("many to 1: target_vols must have only one value, got", target_vols)

      # first target_vols: get or calculate from sources and convert to list of length 1
      target_vols = target_vols if target_vols is not None else self.sum_mixed(source_vol)
      if isinstance(target_vols, numbers.Rational):
        target_vols = [target_vols]

      # source_vol: get or calculate from targets and convert to list (based optionally on ratios)
      if ratios is None:
          ratios = [1] * len(sources)
      if isinstance(source_vol, numbers.Rational):
        source_vol = [source_vol * r / sum(ratios) for r in ratios]
      elif source_vol is None:
        source_vol = [target_vols[0] * r / sum(ratios) for r in ratios]


      # aspiration_flow_rates: convert to list of length sources
      assert aspiration_flow_rates is None or \
        isinstance(aspiration_flow_rates, int) or \
        len(aspiration_flow_rates) == len(sources), \
        ValueError("many to 1: aspiration_flow_rates must be a number or a list with the same length as sources, got", aspiration_flow_rates)
      if isinstance(aspiration_flow_rates, int):
        aspiration_flow_rates = [aspiration_flow_rates] * len(sources)
      # dispense_flow_rates: convert to list of length 1
      assert dispense_flow_rates is None or \
        isinstance(dispense_flow_rates, int) or \
        len(dispense_flow_rates) == len(sources), \
        ValueError("many to 1: dispense_flow_rates must be a number or a list with the same length as sources, got", dispense_flow_rates)
      if isinstance(dispense_flow_rates, int):
        dispense_flow_rates = [dispense_flow_rates] * len(sources)

      # use_channels should be fine
      # todo make 8 dynamic based on size of head
      assert use_channels is None or \
        isinstance(use_channels, int) or \
        (len(use_channels) == len(sources) and len(use_channels) <= 8), \
        ValueError("many to 1: use_channels must have same length as sources and less than 8, got", use_channels)
      if isinstance(use_channels, int):
        use_channels = [use_channels]

      # if done with one tip, first aspirate all sources into a single tip,
      # then dispense the entire volume at once
      if len(use_channels) == 1:
        for source, vol, asp_flow in zip(sources, source_vol, aspiration_flow_rates):
          await self.aspirate(
            resources=[source],
            vols=vol,
            flow_rates=asp_flow,
            use_channels=[use_channels],
            **aspiration_kwargs)
        await self.dispense(
            resources=[targets],
            vols=target_vols,
            flow_rates=dispense_flow_rates[0],
            use_channels=[use_channels],
            **backend_kwargs)
      # if done with multiple tips, aspirate each source into seperate tip
      # then dispense each tip into the same target
      elif len(use_channels) > 1:
        await self.aspirate(
          resources=[sources],
          vols=source_vol,
          flow_rates=aspiration_flow_rates,
          use_channels=[use_channels],
          **aspiration_kwargs)
        for channel, target, vol, disp_flow in zip(use_channels, targets, target_vols, dispense_flow_rates):
          await self.dispense(
            resources=[target],
            vols=vol,
            flow_rates=disp_flow,
            use_channels=[channel],
            **backend_kwargs)
      return

    """ # Case 4: Multiple Wells to Multiple Wells """
    if isinstance(sources, list) and isinstance(targets, list):
      # source_vol, target_vol and ratios:
      assert source_vol is not None or target_vols is not None, \
        ValueError("many to many: Must specify source_vol or target_vols, got", source_vol, target_vols)
      assert source_vol is None or (isinstance(source_vol, list) and len(source_vol) == len(sources)), \
        ValueError("many to many: source_vol must have same length as sources, got", source_vol)
      assert target_vols is None or (isinstance(target_vols, list) and len(target_vols) == len(targets)), \
        ValueError("many to many: target_vols must have same length as targets, got", target_vols)


      """Case 4a: Just Source Volumes"""
      if source_vol is not None and target_vols is None:
        """ Case 4aI: Source Volume is int"""
        if ratios is None:
          ratios = [1] * len(sources)
        # interpret source_vol as total volume
        # if you want to specify volume for each source, pass a list: [vol] * len(sources)
        if isinstance(source_vol, numbers.Rational):
            source_vol = [source_vol / len(sources)] * len(sources)
            target_vols = [source_vol * r / sum(ratios) for r in ratios]
        """ Case 4aII: Source Volume is list"""
        if isinstance(source_vol, list):
          source_vol = source_vol
          target_vols = [sum(source_vol) * r / sum(ratios) for r in ratios]

      """Case 4b: Just Target Volumes"""
      if source_vol is None and target_vols is not None:
        """ Case 4bI: Target Volume is int"""
        if ratios is None:
          ratios = [1] * len(targets)
        # interpret target_vols as total volume
        # if you want to specify volume for each target, pass a list: [vol] * len(targets)
        if isinstance(target_vols, numbers.Rational):
          target_vols = [target_vols / len(targets)] * len(targets)
          source_vol = [target_vols * r / sum(ratios) for r in ratios]

        """ Case 4bII: Target Volume is list"""
        if isinstance(target_vols, list):
          target_vols = target_vols
          source_vol = [sum(target_vols) * r / sum(ratios) for r in ratios]

      """Case 4c: Both Source and Target Volumes"""
      if source_vol is not None and target_vols is not None:
        """ Case 4cI: Source Volume and Target Volume are int"""
        assert not isinstance(source_vol, numbers.Rational) and isinstance(target_vols, numbers.Rational), \
          ValueError("many to many: source_vol and target_vols cannot both be numbers, got", source_vol, target_vols)

        """ Case 4cII: Source Volume and Target Volume is list"""
        if isinstance(source_vol, list) and isinstance(target_vols, list):
          assert sum(source_vol) == sum(target_vols), \
            ValueError("many to many: source_vol and target_vols must have the same sum, got", source_vol, target_vols)
          source_vol = source_vol
          target_vols = target_vols

        """ Case 4cIII: Source Volume is int and Target Volume is list"""
        if isinstance(source_vol, numbers.Rational) and isinstance(target_vols, list):
          assert source_vol == sum(target_vols), \
            ValueError("many to many: source_vol and target_vols must have the same sum, got", source_vol, target_vols)
          # apply ratio to sources
          if ratios is None:
            ratios = [1] * len(sources)
          source_vol = [source_vol * r / sum(ratios) for r in ratios]
          target_vols = target_vols

        """ Case 4cIV: Source Volume is list and Target Volume is int"""
        if isinstance(source_vol, list) and isinstance(target_vols, numbers.Rational):
          assert sum(source_vol) == target_vols, \
            ValueError("many to many: source_vol and target_vols must have the same sum, got", source_vol, target_vols)
          # apply ratio to targets
          if ratios is None:
            ratios = [1] * len(targets)
          source_vol = source_vol
          target_vols = [target_vols * r / sum(ratios) for r in ratios]


      # aspiration_flow_rates: convert to list of length sources
      assert aspiration_flow_rates is None or \
        isinstance(aspiration_flow_rates, int) or \
        (len(aspiration_flow_rates) <= 8 and len(aspiration_flow_rates) == len(sources)), \
        ValueError("many to many: aspiration_flow_rates must be a number or a list with the same length as sources and less than 8, got", aspiration_flow_rates)
      if isinstance(aspiration_flow_rates, int):
        aspiration_flow_rates = [aspiration_flow_rates] * len(sources)

      # convert dispense_flow_rates to list of length targets
      assert dispense_flow_rates is None or \
        isinstance(dispense_flow_rates, int) or \
        (len(dispense_flow_rates) <= 8 and len(dispense_flow_rates) == len(targets)), \
        ValueError("many to many: dispense_flow_rates must be a number or a list with the same length as targets and less than 8, got", dispense_flow_rates)
      if isinstance(dispense_flow_rates, int):
        dispense_flow_rates = [dispense_flow_rates] * len(targets)

      # use channels: length of list indicates if number of tips is directed by targets or sources (default)
      assert use_channels is None or \
        (isinstance(use_channels, list) and \
        (len(use_channels) == len(sources)) or (len(use_channels) == len(targets))), \
        ValueError("many to many: use_channels must have same length as sources or targets, got", use_channels)


      assert round(sum(source_vol), 1) == round(sum(target_vols), 1), \
        ValueError("many to many: something went wrong in the calculation of source and target volumes, got", source_vol, target_vols)
      # now we are ready to aspirate


      if use_channels is None:
        use_channels = list(range(len(sources) if len(sources) <= 8 else 8))
      self._make_sure_channels_exist(use_channels)


      # possible cases
      # [50, 50] to [50, 50] = true
      # [50, 50] to [70, 30] = false, one to large, would require multiple dispenses from different tips into same well
      # [50, 50] to [50, 20, 10] = true, if done with 3 tips
      # [50, 50] to [50, 20, 20, 10] = true, if done with 4 tips
      # [70, 30] to [50, 50] = false, one to large, would require multiple dispenses into
      # [40, 40] to [10, 10, 10, 10, 10, 10, 10, 10] = true, if done with 8 tips
      # [10, 10, 10, 10, 10, 10, 10, 10] to [40, 40] = true, if done with 8 tips

      # ! only allow transforms that can be done with one set of tips

      # Case 4d1: we pipette with tips based on the sources

      print("N to N transfer coming soon")

      """ if len(use_channels) == len(sources):
        # check for any combinations that are not possible
        # we can aspirate all sources into tips in one go
        await self.aspirate(
          resources=sources,
          vols=source_vol,
          flow_rates=aspiration_flow_rates,
          use_channels=use_channels,
          **aspiration_kwargs)

        pass """

      """ number_of_aspirations = 0
      number_of_dispenses = 0
      number_of_tips = len(use_channels)
      asp_channel_combinations = [list(combo) for combo in itertools.combinations(use_channels, len(sources))]
      if len(use_channels) == len(sources):
        number_of_aspirations = 1
        number_of_dispenses = math.ceil(len(targets) / number_of_tips)
        asp_channel_combinations = [use_channels]
      if len(use_channels) == len(targets):
        assert len(use_channels) > len(sources), ValueError(f'many to many: use_channels must be greater than sources, if specified as length targets. Use an array with {len(sources)} instead. Got {use_channels}, {sources}')
        number_of_dispenses = 1
        number_of_aspirations = math.ceil(len(sources) / number_of_tips)
        # we generate a list of combinations of channels to use for each aspiration

      if len(use_channels) == 8:
        number_of_aspirations = math.ceil(len(sources) / number_of_tips)
        number_of_dispenses = math.ceil(len(targets) / number_of_tips)


      for asp in range(number_of_aspirations):
        # Case 1: We aspirate once and dispense many times
        if number_of_aspirations == 1:
          asp_volume = source_vol

        # Case 2: We aspirate many times and dispense once
        if number_of_aspirations > 1:
          asp_volume = [vol / number_of_aspirations for vol in source_vol]

        # Case 3: We aspirate in chunks of 8 and dispense in chunks of 8

        await self.aspirate(
          resources=sources,
          vols=asp_volume,
          flow_rates=aspiration_flow_rates,
          use_channels=list(asp_channel_combinations[asp]),
          **aspiration_kwargs)
      for disp in range(number_of_dispenses):
        if number_of_dispenses == 1:
          disp_volume = target_vols
        await self.dispense(
          resources=targets,
          vols=disp_volume,
          flow_rates=dispense_flow_rates,
          use_channels=use_channels,
          **dispense_kwargs) """
      return


  async def pick_up_tips96(
    self,
    tip_rack: TipRack,
    offset: Coordinate = Coordinate.zero(),
    **backend_kwargs):
    """ Pick up tips using the 96 head. This will pick up 96 tips.

    Examples:
      Pick up tips from a 96-tip tiprack:

      >>> lh.pick_up_tips96(my_tiprack)

    Args:
      tip_rack: The tip rack to pick up tips from.
      offset: The offset to use when picking up tips, optional.
      backend_kwargs: Additional keyword arguments for the backend, optional.
    """

    extras = self._check_args(self.backend.pick_up_tips96, backend_kwargs, default={"pickup"})
    for extra in extras:
      del backend_kwargs[extra]

    await self.backend.pick_up_tips96(
      pickup=PickupTipRack(resource=tip_rack, offset=offset),
      **backend_kwargs
    )

    # Save the tips as picked up.
    self._picked_up_tips96 = tip_rack

  async def drop_tips96(
    self,
    tip_rack: TipRack,
    offset: Coordinate = Coordinate.zero(),
    **backend_kwargs
  ):
    """ Drop tips using the 96 head. This will drop 96 tips.

    Examples:
      Drop tips to a 96-tip tiprack:

      >>> lh.drop_tips96(my_tiprack)

    Args:
      tip_rack: The tip rack to drop tips to.
      offset: The offset to use when dropping tips.
      backend_kwargs: Additional keyword arguments for the backend, optional.
    """

    extras = self._check_args(self.backend.drop_tips96, backend_kwargs, default={"drop"})
    for extra in extras:
      del backend_kwargs[extra]

    await self.backend.drop_tips96(
      drop=DropTipRack(resource=tip_rack, offset=offset),
      **backend_kwargs
    )

    self._picked_up_tips96 = None

  async def return_tips96(self):
    """ Return the tips on the 96 head to the tip rack where they were picked up.

    Examples:
      Return the tips on the 96 head to the tip rack where they were picked up:

      >>> lh.pick_up_tips96(my_tiprack)
      >>> lh.return_tips96()

    Raises:
      RuntimeError: If no tips have been picked up.
    """

    if self._picked_up_tips96 is None:
      raise RuntimeError("No tips have been picked up with the 96 head")

    await self.drop_tips96(self._picked_up_tips96)

  async def aspirate_plate(
    self,
    plate: Plate,
    volume: float,
    flow_rate: Optional[float] = None,
    end_delay: float = 0,
    **backend_kwargs
  ):
    """ Aspirate from all wells in a plate.

    Examples:
      Aspirate an entire 96 well plate:

      >>> lh.aspirate_plate(plate, volume=50)

    Args:
      resource: Resource name or resource object.
      pattern: Either a list of lists of booleans where inner lists represent rows and outer lists
        represent columns, or a string representing a range of positions. Default all.
      volume: The volume to aspirate from each well.
      flow_rate: The flow rate to use when aspirating, in ul/s. If `None`, the backend default
        will be used.
      end_delay: The delay after the last aspiration in seconds, optional. This is useful for when
        the tips used in the aspiration are dripping.
      backend_kwargs: Additional keyword arguments for the backend, optional.
    """

    extras = self._check_args(self.backend.aspirate96, backend_kwargs, default={"aspiration"})
    for extra in extras:
      del backend_kwargs[extra]

    if self._picked_up_tips96 is None:
      raise NoTipError("No tips have been picked up with the 96 head")
    tips = self._picked_up_tips96.get_all_tips()

    if plate.has_lid():
      raise ValueError("Aspirating from plate with lid")

    # liquid(s) for each channel. If volume tracking is disabled, use None as the liquid.
    liquids: List[List[Tuple[Optional[Liquid], float]]] = []
    for w in plate.get_all_items():
      if w.tracker.is_disabled or not does_volume_tracking():
        liquids.append([(None, volume)])
      else:
        liquids.append(w.tracker.get_liquids(top_volume=volume))

    if plate.num_items_x == 12 and plate.num_items_y == 8:
      await self.backend.aspirate96(aspiration=AspirationPlate(
        resource=plate,
        volume=volume,
        offset=Coordinate.zero(),
        flow_rate=flow_rate,
        tips=tips,
        liquid_height=None,
        blow_out_air_volume=0,
        liquids=liquids),
        **backend_kwargs)
    else:
      raise NotImplementedError(f"It is not possible to plate aspirate from an {plate.num_items_x} "
                                f"by {plate.num_items_y} plate")

    if end_delay > 0:
      time.sleep(end_delay)

  async def dispense_plate(
    self,
    plate: Plate,
    volume: float,
    flow_rate: Optional[float] = None,
    end_delay: float = 0,
    **backend_kwargs
  ):
    """ Dispense to all wells in a plate.

    Examples:
      Dispense an entire 96 well plate:

      >>> dispense96(plate, volume=50)

    Args:
      resource: Resource name or resource object.
      pattern: Either a list of lists of booleans where inner lists represent rows and outer lists
        represent columns, or a string representing a range of positions. Default all.
      volume: The volume to dispense to each well.
      flow_rate: The flow rate to use when aspirating, in ul/s. If `None`, the backend default
        will be used.
      end_delay: The delay after the last dispense in seconds, optional. This is useful for when
        the tips used in the dispense are dripping.
      backend_kwargs: Additional keyword arguments for the backend, optional.
    """

    extras = self._check_args(self.backend.dispense96, backend_kwargs, default={"dispense"})
    for extra in extras:
      del backend_kwargs[extra]

    if self._picked_up_tips96 is None:
      raise NoTipError("No tips have been picked up with the 96 head")
    tips = self._picked_up_tips96.get_all_tips()

    if plate.has_lid():
      raise ValueError("Dispensing to plate with lid")

    # liquid(s) for each channel. If volume tracking is disabled, use None as the liquid.
    liquids: List[List[Tuple[Optional[Liquid], float]]] = []
    for w in plate.get_all_items():
      if w.tracker.is_disabled or not does_volume_tracking():
        liquids.append([(None, volume)])
      else:
        liquids.append(w.tracker.get_liquids(top_volume=volume))

    if plate.num_items_x == 12 and plate.num_items_y == 8:
      await self.backend.dispense96(dispense=DispensePlate(
        resource=plate,
        volume=volume,
        offset=Coordinate.zero(),
        flow_rate=flow_rate,
        tips=tips,
        liquid_height=None,
        blow_out_air_volume=0,
        liquids=liquids),
        **backend_kwargs)
    else:
      raise NotImplementedError(f"It is not possible to plate dispense to an {plate.num_items_x} "
                               f"by {plate.num_items_y} plate")

    if end_delay > 0:
      time.sleep(end_delay)

  async def stamp(
    self,
    source: Plate,
    target: Plate,
    volume: float,
    aspiration_flow_rate: Optional[float] = None,
    dispense_flow_rate: Optional[float] = None,
  ):
    """ Stamp (aspiration and dispense) one plate onto another.

    Args:
      source: the source plate
      target: the target plate
      volume: the volume to be transported
      aspiration_flow_rate: the flow rate for the aspiration, in ul/s. If `None`, the backend
        default will be used.
      dispense_flow_rate: the flow rate for the dispense, in ul/s. If `None`, the backend default
        will be used.
    """

    assert (source.num_items_x, source.num_items_y) == (target.num_items_x, target.num_items_y), \
      "Source and target plates must be the same shape"

    await self.aspirate_plate(
      plate=source,
      volume=volume,
      flow_rate=aspiration_flow_rate)
    await self.dispense_plate(
      plate=source,
      volume=volume,
      flow_rate=dispense_flow_rate)

  async def move_resource(
    self,
    resource: Resource,
    to: Coordinate,
    intermediate_locations: Optional[List[Coordinate]] = None,
    resource_offset: Coordinate = Coordinate.zero(),
    destination_offset: Coordinate = Coordinate.zero(),
    pickup_distance_from_top: float = 0,
    get_direction: GripDirection = GripDirection.FRONT,
    put_direction: GripDirection = GripDirection.FRONT,
    **backend_kwargs
  ):
    """ Move a resource to a new location.

    Has convenience methods :meth:`move_plate` and :meth:`move_lid`.

    Examples:
      Move a plate to a new location:

      >>> lh.move_resource(plate, to=Coordinate(100, 100, 100))

    Args:
      resource: The Resource object.
      to: The absolute coordinate (meaning relative to deck) to move the resource to.
      intermediate_locations: A list of intermediate locations to move the resource through.
      resource_offset: The offset from the resource's origin, optional (rarely necessary).
      destination_offset: The offset from the location's origin, optional (rarely necessary).
      pickup_distance_from_top: The distance from the top of the resource to pick up from.
      get_direction: The direction from which to pick up the resource.
      put_direction: The direction from which to put down the resource.
    """

    extras = self._check_args(self.backend.move_resource, backend_kwargs, default={"move"})
    for extra in extras:
      del backend_kwargs[extra]

    return await self.backend.move_resource(move=Move(
      resource=resource,
      destination=to,
      intermediate_locations=intermediate_locations or [],
      resource_offset=resource_offset,
      destination_offset=destination_offset,
      pickup_distance_from_top=pickup_distance_from_top,
      get_direction=get_direction,
      put_direction=put_direction),
      **backend_kwargs)

  async def move_lid(
    self,
    lid: Lid,
    to: Union[Plate, ResourceStack, Coordinate],
    intermediate_locations: Optional[List[Coordinate]] = None,
    resource_offset: Coordinate = Coordinate.zero(),
    to_offset: Coordinate = Coordinate.zero(),
    get_direction: GripDirection = GripDirection.FRONT,
    put_direction: GripDirection = GripDirection.FRONT,
    pickup_distance_from_top: float = 5.7,
    **backend_kwargs
  ):
    """ Move a lid to a new location.

    A convenience method for :meth:`move_resource`.

    Examples:
      Move a lid to the :class:`~resources.ResourceStack`:

      >>> lh.move_lid(plate.lid, stacking_area)

      Move a lid to the stacking area and back, grabbing it from the left side:

      >>> lh.move_lid(plate.lid, stacking_area, get_direction=GripDirection.LEFT)
      >>> lh.move_lid(stacking_area.get_top_item(), plate, put_direction=GripDirection.LEFT)

    Args:
      lid: The lid to move. Can be either a Plate object or a Lid object.
      to: The location to move the lid to, either a plate, ResourceStack or a Coordinate.
      resource_offset: The offset from the resource's origin, optional (rarely necessary).
      to_offset: The offset from the location's origin, optional (rarely necessary).

    Raises:
      ValueError: If the lid is not assigned to a resource.
    """

    if isinstance(to, Plate):
      to_location = to.get_absolute_location()
      to_location = Coordinate(
        x=to_location.x,
        y=to_location.y,
        z=to_location.z  + to.get_size_z() - lid.get_size_z())
    elif isinstance(to, ResourceStack):
      assert to.direction == "z", "Only ResourceStacks with direction 'z' are currently supported"
      to_location = to.get_absolute_location()
      to_location = Coordinate(
        x=to_location.x,
        y=to_location.y,
        z=to_location.z  + to.get_size_z())
    elif isinstance(to, Coordinate):
      to_location = to
    else:
      raise ValueError(f"Cannot move lid to {to}")

    await self.move_resource(
      lid,
      to=to_location,
      intermediate_locations=intermediate_locations,
      pickup_distance_from_top=pickup_distance_from_top,
      resource_offset=resource_offset,
      to_offset=to_offset,
      get_direction=get_direction,
      put_direction=put_direction,
      **backend_kwargs)

    lid.unassign()
    if isinstance(to, Coordinate):
      self.deck.assign_child_resource(lid, location=to_location)
    elif isinstance(to, ResourceStack): # manage its own resources
      to.assign_child_resource(lid)
    else:
      to.assign_child_resource(lid, location=to_location)

  async def move_plate(
    self,
    plate: Plate,
    to: Union[ResourceStack, CarrierSite, Resource, Coordinate],
    intermediate_locations: Optional[List[Coordinate]] = None,
    resource_offset: Coordinate = Coordinate.zero(),
    to_offset: Coordinate = Coordinate.zero(),
    put_direction: GripDirection = GripDirection.FRONT,
    get_direction: GripDirection = GripDirection.FRONT,
    pickup_distance_from_top: float = 13.2,
    **backend_kwargs
  ):
    """ Move a plate to a new location.

    A convenience method for :meth:`move_resource`.

    Examples:
      Move a plate to into a carrier spot:

      >>> lh.move_plate(plate, plt_car[1])

      Move a plate to an absolute location:

      >>> lh.move_plate(plate_01, Coordinate(100, 100, 100))

      Move a lid to another carrier spot, grabbing it from the left side:

      >>> lh.move_plate(plate, plt_car[1], get_direction=GripDirection.LEFT)
      >>> lh.move_plate(plate, plt_car[0], put_direction=GripDirection.LEFT)

      Move a resource while visiting a few intermediate locations along the way:

      >>> lh.move_plate(plate, plt_car[1], intermediate_locations=[
      ...   Coordinate(100, 100, 100),
      ...   Coordinate(200, 200, 200),
      ... ])

    Args:
      plate: The plate to move. Can be either a Plate object or a CarrierSite object.
      to: The location to move the plate to, either a plate, CarrierSite or a Coordinate.
      resource_offset: The offset from the resource's origin, optional (rarely necessary).
      to_offset: The offset from the location's origin, optional (rarely necessary).
    """

    if isinstance(to, ResourceStack):
      assert to.direction == "z", "Only ResourceStacks with direction 'z' are currently supported"
      to_location = to.get_absolute_location()
      to_location = Coordinate(
        x=to_location.x,
        y=to_location.y,
        z=to_location.z  + to.get_size_z())
    elif isinstance(to, Coordinate):
      to_location = to
    else:
      to_location = to.get_absolute_location()

    await self.move_resource(
      plate,
      to=to_location,
      intermediate_locations=intermediate_locations,
      pickup_distance_from_top=pickup_distance_from_top,
      resource_offset=resource_offset,
      to_offset=to_offset,
      get_direction=get_direction,
      put_direction=put_direction,
      **backend_kwargs)

    plate.unassign()
    if isinstance(to, Coordinate):
      to_location -= self.deck.location # passed as an absolute location, but stored as relative
      self.deck.assign_child_resource(plate, location=to_location)
    elif isinstance(to, CarrierSite): # .zero() resources
      to.assign_child_resource(plate, location=Coordinate.zero())
    elif isinstance(to, (ResourceStack, PlateReader)): # manage its own resources
      to.assign_child_resource(plate)
    else:
      to.assign_child_resource(plate, location=to_location)

  def serialize(self) -> dict:
    """ Serialize the liquid handler to a dictionary.

    Returns:
      A dictionary representation of the liquid handler.
    """

    return {
      "deck": self.deck.serialize(),
      "backend": self.backend.serialize()
    }

  def save(self, path: str):
    """ Save the liquid handler to a file.

    Args:
      path: The path to the file to save to.
    """

    with open(path, "w", encoding="utf-8") as f:
      json.dump(self.serialize(), f, indent=2)

  @classmethod
  def deserialize(cls, data: dict) -> LiquidHandler:
    """ Deserialize a liquid handler from a dictionary.

    Args:
      data: A dictionary representation of the liquid handler.
    """

    deck = Deck.deserialize(data["deck"])
    backend = LiquidHandlerBackend.deserialize(data["backend"])
    return LiquidHandler(deck=deck, backend=backend)

  @classmethod
  def load(cls, path: str) -> LiquidHandler:
    """ Load a liquid handler from a file.

    Args:
      path: The path to the file to load from.
    """

    with open(path, "r", encoding="utf-8") as f:
      return cls.deserialize(json.load(f))
