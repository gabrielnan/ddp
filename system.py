from abc import ABCMeta, abstractmethod


class System(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def F(self, x, u):
        """Dynamics function (dx/dt)

        :param x: state
        :param u: control
        :return: dx/dt
        """
        return

