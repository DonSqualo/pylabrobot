Welcome to PyLabRobot's documentation!
======================================

PyLabRobot is a hardware agnostic, pure Python library for liquid handling robots and accessories.

PyLabRobot provides a layer of general-purpose abstractions over robot functions, with various device drivers for communicating with different kinds of robots. Right now we only support Hamilton STAR and STARLet, Tecan EVO, and Opentrons robots, but we will soon support many more. All of these robots can be controlled using any computer running any operating system. We also provide a simulator which simulates protocols in a browser-based deck visualization.

- GitHub repository: https://github.com/PyLabRobot/pylabrobot
- Forum: https://forums.pylabrobot.org
- Paper: https://www.cell.com/device/fulltext/S2666-9986(23)00170-9

.. image:: img/plr.jpg
  :width: 600
  :alt: Graphical abstract of PyLabRobot

.. note::
  PyLabRobot is different from `PyHamilton <https://github.com/dgretton/pyhamilton>`_. While both packages are created by the same lab and both provide a Python interfaces to Hamilton robots, PyLabRobot aims to provide a universal interface to many different robots runnable on many different computers, where PyHamilton is a Windows only interface to Hamilton's VENUS.


.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   installation.md
   contributing.md
   how-to-open-source.md


.. toctree::
   :maxdepth: 1
   :caption: Liquid handling

   basic
   using-the-simulator
   using-trackers
   custom-resources
   writing-robot-agnostic-methods


.. toctree::
   :maxdepth: 1
   :caption: Plate reading

   plate_reading


.. toctree::
   :maxdepth: 1
   :caption: Pumps

   pumps


.. toctree::
   :maxdepth: 4
   :caption: API documentation

   pylabrobot


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


