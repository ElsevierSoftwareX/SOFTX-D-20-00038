class Plural:
    __slots__ = 'word', 'value', 'singular', 'plural', 'zero', 'ignore_negatives'

    def __init__(self, value: int, word: str = "", **kwargs):
        """
        Parameters
        ----------
        value : int
            The determining value
        word : str, optional
            The word to make plural. (defaults to "")
        singular : str, optional
            Appended to `word` if `value` == 1. (defaults to '')
        plural : str, optional
            Appended to `word` if `value` > 1. (defaults to 's')
        zero : str, optional
            Replaces `value` if `value` == 0. (defaults to 0)
        ignore_negatives : bool, optional
            This will raise ValueError if `value` is negative. (defaults to False)
            Set to True if you don't care about negative values.
        """

        self.value, self.word = value, word
        self.singular = kwargs.pop('singular', '')
        self.plural = kwargs.pop('plural', 's')
        self.zero = kwargs.pop('zero', 0)
        self.ignore_negatives = kwargs.pop('ignore_negatives', False)

    def __str__(self):
        v = self.value
        pluralizer = self.plural if abs(v) > 1 else self.singular

        if v < 0 and not self.ignore_negatives:
            raise ValueError

        return f"{v or self.zero} {self.word}{pluralizer}"
