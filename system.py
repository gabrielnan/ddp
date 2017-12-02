from abc import ABCMeta, abstractmethod


class System(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        self.x_names = None

    @abstractmethod
    def F(self, x, u):
        """Dynamics function (dx/dt)

        :param x: state
        :param u: control
        :return: dx/dt
        """
        return

