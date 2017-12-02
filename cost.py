from abc import ABCMeta, abstractmethod


class Cost(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def lagr(self, x):
        """Running cost function (Lagrangian)

        :param x: state
        :return: running cost
        """
        return

    @abstractmethod
    def phi(self, x):
        """Terminal cost function

        :param x: state
        :return: terminal cost for state x
        """
        return